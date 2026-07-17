"""Packs a trained model plus its configs/stats into a distributable archive.

`pack()` bundles arbitrary files (checkpoints, stats) and YAML configs into a
single tar or zip archive alongside a generated `meta.yaml` manifest;
`unpack()` reverses this, extracting the archive and rewriting path
references inside the YAML configs to point at the extraction directory.
`Archiver` is a small tar/zip-format-agnostic wrapper used by both.

The on-disk archive layout (a `meta.yaml` manifest listing `yaml_files` and
`files`, plus the files themselves at their original relative/resolved
paths) and the archive formats supported by file suffix (`.tar`, `.tgz`/
`.tar.gz`, `.tbz2`/`.tar.bz2`, `.txz`/`.tar.xz`, `.zip`) are an on-disk
contract with previously-packed artifacts and must not change.
"""

import os
import sys
import tarfile
import zipfile
from datetime import datetime
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Any, Dict, IO, Iterable, Iterator, Optional, Tuple, Union

import yaml

ArchiveInfo = Union[tarfile.TarInfo, zipfile.ZipInfo]


def _detect_archive_type_and_mode(
    file: Union[str, Path], mode: str
) -> Tuple[str, str]:
    """Infer archive type ("tar"/"zip") and the effective open-mode from a filename.

    For compressed tar variants (`.tgz`, `.tbz2`, `.txz`) opened for writing,
    the plain `"w"` mode is upgraded to tarfile's compression-specific mode
    (`"w:gz"`, `"w:bz2"`, `"w:xz"`) so `tarfile.open` selects the matching
    compressor; other modes (e.g. `"r"`) are left untouched since `tarfile`
    auto-detects compression on read.

    Raises:
        ValueError: If the file suffix doesn't match a supported archive type.
    """
    suffix = Path(file).suffix
    suffixes = Path(file).suffixes
    if suffix == ".tar":
        return "tar", mode
    elif suffix == ".tgz" or suffixes == [".tar", ".gz"]:
        return "tar", ("w:gz" if mode == "w" else mode)
    elif suffix == ".tbz2" or suffixes == [".tar", ".bz2"]:
        return "tar", ("w:bz2" if mode == "w" else mode)
    elif suffix == ".txz" or suffixes == [".tar", ".xz"]:
        return "tar", ("w:xz" if mode == "w" else mode)
    elif suffix == ".zip":
        return "zip", mode
    else:
        raise ValueError(f"Cannot detect archive format: type={file}")


class Archiver:
    """Tar/zip-format-agnostic wrapper exposing a small, shared archive API.

    The concrete archive type is inferred from `file`'s suffix (see
    `_detect_archive_type_and_mode`) and stored in `self.type` ("tar" or
    "zip"); every method dispatches on `self.type` to the matching
    `tarfile`/`zipfile` call so callers don't need to know which backend
    is in use.
    """

    def __init__(self, file: Union[str, Path], mode: str = "r"):
        """Open `file` as a tar or zip archive, inferring the format from its suffix.

        Args:
            file: Path to the archive to read or create.
            mode: Open mode, e.g. `"r"` or `"w"` (passed through to
                `tarfile.open`/`zipfile.ZipFile`, with compression-specific
                tar write-modes chosen automatically based on `file`'s suffix).

        Raises:
            ValueError: If the archive format can't be inferred from `file`.
        """
        self.type, mode = _detect_archive_type_and_mode(file, mode)

        if self.type == "tar":
            self.fopen = tarfile.open(file, mode=mode)
        elif self.type == "zip":
            self.fopen = zipfile.ZipFile(file, mode=mode)
        else:
            raise ValueError(f"Not supported: type={type}")

    def __enter__(self) -> "Archiver":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.fopen.close()

    def close(self) -> None:
        """Close the underlying archive file handle."""
        self.fopen.close()

    def __iter__(self) -> Iterator[ArchiveInfo]:
        """Iterate over archive member info objects (`TarInfo`/`ZipInfo`)."""
        if self.type == "tar":
            return iter(self.fopen)
        elif self.type == "zip":
            return iter(self.fopen.infolist())
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def add(
        self,
        filename: Union[str, Path],
        arcname: Optional[Union[str, Path]] = None,
        recursive: bool = True,
    ):
        """Add a file (or, recursively, a directory) to the archive.

        Args:
            filename: Path on disk to add.
            arcname: Name to store the file under in the archive; defaults
                to `filename` itself.
            recursive: If True and `filename` is a directory, add every file
                under it (preserving relative paths under `arcname`) instead
                of raising/skipping.
        """
        if arcname is not None:
            print(f"adding: {arcname}")
        else:
            print(f"adding: {filename}")

        if recursive and Path(filename).is_dir():
            for f in Path(filename).glob("**/*"):
                if f.is_dir():
                    continue

                if arcname is not None:
                    _arcname = Path(arcname) / f
                else:
                    _arcname = None

                self.add(f, _arcname)
            return

        if self.type == "tar":
            return self.fopen.add(filename, arcname)
        elif self.type == "zip":
            return self.fopen.write(filename, arcname)
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def addfile(self, info: ArchiveInfo, fileobj: IO[bytes]):
        """Write `fileobj`'s contents into the archive as the member `info`."""
        print(f"adding: {self.get_name_from_info(info)}")

        if self.type == "tar":
            return self.fopen.addfile(info, fileobj)
        elif self.type == "zip":
            return self.fopen.writestr(info, fileobj.read())
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def generate_info(self, name: Union[str, Path], size: int) -> ArchiveInfo:
        """Generate TarInfo using system information"""
        if self.type == "tar":
            tarinfo = tarfile.TarInfo(str(name))
            if os.name == "posix":
                tarinfo.gid = os.getgid()
                tarinfo.uid = os.getuid()
            tarinfo.mtime = datetime.now().timestamp()
            tarinfo.size = size
            # Keep mode as default
            return tarinfo
        elif self.type == "zip":
            zipinfo = zipfile.ZipInfo(str(name), datetime.now().timetuple()[:6])
            zipinfo.file_size = size
            return zipinfo
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def get_name_from_info(self, info: ArchiveInfo) -> str:
        """Return the archive-relative member name/path stored in `info`."""
        if self.type == "tar":
            assert isinstance(info, tarfile.TarInfo), type(info)
            return info.name
        elif self.type == "zip":
            assert isinstance(info, zipfile.ZipInfo), type(info)
            return info.filename
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def extract(self, info: ArchiveInfo, path: Optional[Union[str, Path]] = None):
        """Extract the archive member `info` to `path` (defaults to cwd)."""
        if self.type == "tar":
            return self.fopen.extract(info, path)
        elif self.type == "zip":
            return self.fopen.extract(info, path)
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def extractfile(self, info: ArchiveInfo, mode: str = "r") -> IO:
        """Open the archive member `info` for reading.

        Args:
            info: Archive member to open.
            mode: `"r"` returns a text-mode file object; any other value
                (e.g. `"rb"`) returns a binary-mode file object.
        """
        if self.type == "tar":
            f = self.fopen.extractfile(info)
            if mode == "r":
                return TextIOWrapper(f)
            else:
                return f
        elif self.type == "zip":
            if mode == "rb":
                mode = "r"
            return self.fopen.open(info, mode)
        else:
            raise ValueError(f"Not supported: type={self.type}")


def find_path_and_change_it_recursive(value: Any, src: str, tgt: str) -> Any:
    """Recursively replace any string equal to path `src` with `tgt`.

    Walks `value` (which may be an arbitrarily nested combination of dicts,
    lists, and tuples, as produced by `yaml.safe_load`) and returns an
    equivalent structure where every string leaf that resolves to the same
    path as `src` is replaced with `tgt`. Used to rewrite path references
    inside a packed archive's YAML configs after extraction.

    Args:
        value: A YAML-loaded value (dict/list/tuple/str/other scalar).
        src: The path to search for (compared via `Path(value) == Path(src)`).
        tgt: The replacement string.

    Returns:
        A new structure with matching path strings replaced; non-matching
        values are returned as-is (lists/tuples become new lists).
    """
    if isinstance(value, dict):
        return {
            k: find_path_and_change_it_recursive(v, src, tgt) for k, v in value.items()
        }
    elif isinstance(value, (list, tuple)):
        return [find_path_and_change_it_recursive(v, src, tgt) for v in value]
    elif isinstance(value, str) and Path(value) == Path(src):
        return tgt
    else:
        return value


def get_dict_from_cache(meta: Union[Path, str]) -> Optional[Dict[str, str]]:
    """Read a previously-extracted `meta.yaml` and validate its files still exist.

    Args:
        meta: Path to the extracted `meta.yaml`, located at
            `{outpath}/{some_subdir}/meta.yaml`; `outpath` is derived as
            `meta.parent.parent`.

    Returns:
        A dict mapping each `yaml_files`/`files` key to its resolved path
        under `outpath`, or None if `meta` doesn't exist or any referenced
        file is missing (i.e. the cache is stale/incomplete).
    """
    meta = Path(meta)
    outpath = meta.parent.parent
    if not meta.exists():
        return None

    with meta.open("r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
        assert isinstance(d, dict), type(d)
        yaml_files = d["yaml_files"]
        files = d["files"]
        assert isinstance(yaml_files, dict), type(yaml_files)
        assert isinstance(files, dict), type(files)

        retval = {}
        for key, value in list(yaml_files.items()) + list(files.items()):
            if not (outpath / value).exists():
                return None
            retval[key] = str(outpath / value)
        return retval


def unpack(
    input_archive: Union[Path, str],
    outpath: Union[Path, str],
    use_cache: bool = True,
) -> Dict[str, str]:
    """Scan all files in the archive file and return as a dict of files.

    Reads the archive's `meta.yaml` manifest, extracts every member under
    `outpath`, and rewrites path references inside the extracted YAML config
    files (`yaml_files` in the manifest) to point at their new location
    under `outpath` instead of their original packing-time paths.

    Args:
        input_archive: Path to the `.tar`/`.tgz`/`.tbz2`/`.txz`/`.zip`
            archive produced by `pack()`.
        outpath: Directory to extract the archive into.
        use_cache: If True and `outpath` already contains a `meta.yaml` from
            a prior extraction (with all referenced files still present),
            skip re-extracting and reuse that cached result.

    Returns:
        Dict mapping each `yaml_files`/`files` key from the manifest to its
        extracted path under `outpath`.

    Raises:
        RuntimeError: If the archive doesn't contain a `meta.yaml` member.

    Examples:
        tarfile:
           model.pth
           some1.file
           some2.file

        >>> unpack("tarfile", "out")
        {'asr_model_file': 'out/model.pth'}
    """
    input_archive = Path(input_archive)
    outpath = Path(outpath)

    with Archiver(input_archive) as archive:
        for info in archive:
            if Path(archive.get_name_from_info(info)).name == "meta.yaml":
                if (
                    use_cache
                    and (outpath / Path(archive.get_name_from_info(info))).exists()
                ):
                    retval = get_dict_from_cache(
                        outpath / Path(archive.get_name_from_info(info))
                    )
                    if retval is not None:
                        return retval
                d = yaml.safe_load(archive.extractfile(info))
                assert isinstance(d, dict), type(d)
                yaml_files = d["yaml_files"]
                files = d["files"]
                assert isinstance(yaml_files, dict), type(yaml_files)
                assert isinstance(files, dict), type(files)
                break
        else:
            raise RuntimeError("Format error: not found meta.yaml")

        for info in archive:
            fname = archive.get_name_from_info(info)
            outname = outpath / fname
            outname.parent.mkdir(parents=True, exist_ok=True)
            if fname in set(yaml_files.values()):
                d = yaml.safe_load(archive.extractfile(info))
                # Rewrite yaml
                for info2 in archive:
                    name = archive.get_name_from_info(info2)
                    d = find_path_and_change_it_recursive(d, name, str(outpath / name))
                with outname.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(d, f)
            else:
                archive.extract(info, path=outpath)

        retval = {}
        for key, value in list(yaml_files.items()) + list(files.items()):
            retval[key] = str(outpath / value)
        return retval


def _to_relative_or_resolve(f: Union[str, Path]) -> str:
    """Resolve `f` to an absolute path (following symlinks), then make it
    relative to the current working directory if possible.

    Returns:
        The relative path as a string if `f` lives under the cwd, otherwise
        the resolved absolute path.
    """
    # Resolve to avoid symbolic link
    p = Path(f).resolve()
    try:
        # Change to relative if it can
        p = p.relative_to(Path(".").resolve())
    except ValueError:
        pass
    return str(p)


def pack(
    files: Dict[str, Union[str, Path]],
    yaml_files: Dict[str, Union[str, Path]],
    outpath: Union[str, Path],
    option: Iterable[Union[str, Path]] = (),
) -> None:
    """Bundle files and YAML configs into a single archive with a `meta.yaml` manifest.

    Args:
        files: Non-YAML files to include (e.g. model checkpoints, stats),
            keyed by a caller-chosen name (e.g. `"model_file"`) that is
            recorded in `meta.yaml` and later returned by `unpack()`.
        yaml_files: YAML config files to include, keyed the same way.
        outpath: Destination archive path; its format is inferred from its
            suffix (see `Archiver`/`_detect_archive_type_and_mode`). Parent
            directories are created if needed.
        option: Extra files to include without a manifest key (e.g.
            supplementary assets referenced by paths *within* the packed
            YAML configs).

    Raises:
        FileNotFoundError: If any path in `files`, `yaml_files`, or `option`
            doesn't exist.

    Side Effects:
        Writes the archive to `outpath` and prints each file as it is added.
    """
    for v in list(files.values()) + list(yaml_files.values()) + list(option):
        if not Path(v).exists():
            raise FileNotFoundError(f"No such file or directory: {v}")

    files = {k: _to_relative_or_resolve(v) for k, v in files.items()}
    yaml_files = {k: _to_relative_or_resolve(v) for k, v in yaml_files.items()}
    option = [_to_relative_or_resolve(v) for v in option]

    meta_objs = dict(
        files=files,
        yaml_files=yaml_files,
        timestamp=datetime.now().timestamp(),
        python=sys.version,
    )

    try:
        import torch

        meta_objs.update(torch=str(torch.__version__))
    except ImportError:
        pass
    try:
        import espnet

        meta_objs.update(espnet=espnet.__version__)
    except ImportError:
        pass

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with Archiver(outpath, mode="w") as archive:
        # Write packed/meta.yaml
        fileobj = BytesIO(yaml.safe_dump(meta_objs).encode())
        info = archive.generate_info("meta.yaml", fileobj.getbuffer().nbytes)
        archive.addfile(info, fileobj=fileobj)

        for f in list(yaml_files.values()) + list(files.values()) + list(option):
            archive.add(f)

    print(f"Generate: {outpath}")
