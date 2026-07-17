"""Readers/writer for the MSVS (multi-singer) JSON dataset metadata format.

Each dataset directory holds one or more JSON files (``train.json``,
``valid*.json``, ``test.json``) containing a list of utterance metadata
dicts, e.g.::

    [
        {"Uid": "utterance_id_A", "Duration": 3.2, "Path": "/path/a.wav"},
        {"Uid": "utterance_id_B", "Duration": 4.1, "Path": "/path/b.wav"},
        ...
    ]

A top-level YAML config then points at one or more of these dataset
directories, e.g.::

    datasetsdirpath: /path/to/datasets
    singing_datasets: [dataset1, dataset2]
    valid_datasets: [dataset3]
    test_datasets: [dataset4]

``read_msvs_json_from_yaml`` and friends flatten these into a single
``{uid: value}`` dict, similar in spirit to a Kaldi scp file.
"""

import collections.abc
import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import yaml
from typeguard import check_argument_types


class NpyMSVSJsonWriter:
    """Writer that saves numpy arrays and records their paths in a scp file.

    Writes one ``{key}.npy`` file per entry under ``outdir``, and appends a
    line ``"{key} {path}\\n"`` to ``scpfile`` for each one, e.g.::

        key1 /some/path/a.npy
        key2 /some/path/b.npy
        key3 /some/path/c.npy
        key4 /some/path/d.npy
        ...

    Examples:
        >>> writer = NpyMSVSJsonWriter('./data/', './data/feat.scp')
        >>> writer['aa'] = numpy_array
        >>> writer['bb'] = numpy_array

    """

    def __init__(self, outdir: Union[Path, str], scpfile: Union[Path, str]):
        assert check_argument_types()
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")

        self.data: Dict[str, str] = {}

    def get_path(self, key: str) -> str:
        """Return the on-disk ``.npy`` path previously written for ``key``."""
        return self.data[key]

    def __setitem__(self, key: str, value: np.ndarray):
        assert isinstance(value, np.ndarray), type(value)
        npy_path = self.dir / f"{key}.npy"
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_path), value)
        self.fscp.write(f"{key} {npy_path}\n")

        # Store the file path
        self.data[key] = str(npy_path)

    def __enter__(self) -> "NpyMSVSJsonWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()


def _load_yaml_config(path: Union[Path, str]) -> dict:
    """Load a YAML config file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_msvs_json_from_yaml(
    path: Union[Path, str], read_type: str = "text"
) -> Dict[str, str]:
    """Dispatch to the train/valid/test MSVS json readers based on ``path``.

    ``path`` may optionally be prefixed with ``"{data_type}#"`` (one of
    ``train``, ``valid``, ``test``) to select which split's config keys to
    read from the YAML file; if no prefix is given, ``"train"`` is assumed.

    Args:
        path: Either ``"{data_type}#{yaml_path}"`` or plain ``yaml_path``.
        read_type: Forwarded to the per-split reader (``"text"``, ``"npy"``,
            ``"wav"`` or ``"dict"`` depending on the split).

    Returns:
        Mapping from utterance id to the value selected by ``read_type``.
    """
    data_type, path = path.split("#", 1) if "#" in path else ("train", path)

    if data_type == "train":
        config_data = _load_yaml_config(path)
        datasetsdirpath = Path(config_data["datasetsdirpath"])
        singing_datasets = config_data["singing_datasets"]
        return read_msvs_json_train(datasetsdirpath, singing_datasets, read_type)
    elif data_type == "valid":
        config_data = _load_yaml_config(path)
        datasetsdirpath = Path(config_data["datasetsdirpath"])
        valid_datasets = config_data["valid_datasets"]
        return read_msvs_json_valid(datasetsdirpath, valid_datasets, read_type)
    elif data_type == "test":
        if "#" in path:
            test_data_path, yaml_path = path.split("#", 1)
            config_data = _load_yaml_config(yaml_path)
        else:
            config_data = _load_yaml_config(path)
            test_data_path = config_data["test_datasets"]
        datasetsdirpath = Path(config_data["datasetsdirpath"])
        return read_msvs_json_test(datasetsdirpath, test_data_path, read_type)


def read_msvs_json_train(
    datasetsdirpath: Union[Path, str],
    singing_datasets: List[str],
    read_type: str = "text",
) -> Dict[str, str]:
    """Read ``{dataset}/train.json`` for each dataset and flatten to a dict.

    Args:
        datasetsdirpath: Root directory containing one subdirectory per
            dataset.
        singing_datasets: Names of the dataset subdirectories to read.
        read_type: ``"text"`` maps each uid to ``""``; ``"npy"`` maps to the
            expected ``{dataset}/audios/{uid}.npy`` path; ``"wav"`` maps to
            the ``Path`` field from the JSON metadata. Utterances shorter
            than 2.0 seconds are skipped.

    Returns:
        Mapping from utterance id to the value selected by ``read_type``.
    """
    if isinstance(datasetsdirpath, str):
        datasetsdirpath = Path(datasetsdirpath)
    data: Dict[str, str] = {}
    for singing_dataset in singing_datasets:
        dataset_json_path = datasetsdirpath / singing_dataset / "train.json"
        with open(dataset_json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        for utt_data in metadata:
            duration = float(utt_data["Duration"])
            if duration < 2.0:
                continue
            if read_type == "text":
                key, value = utt_data["Uid"], ""
            elif read_type == "npy":
                key, value = (
                    utt_data["Uid"],
                    str(
                        datasetsdirpath
                        / singing_dataset
                        / "audios"
                        / f"{utt_data['Uid']}.npy"
                    ),
                )
            elif read_type == "wav":
                key, value = utt_data["Uid"], utt_data["Path"]

            if key in data:
                raise RuntimeError(
                    f"{key} is duplicated ({dataset_json_path}:{utt_data['Uid']})"
                )

            data[key] = value
    return data


def read_msvs_json_valid(
    datasetsdirpath: Union[Path, str],
    valid_datasets: List[str],
    read_type: str = "text",
) -> Dict[str, str]:
    """Read every ``{dataset}/valid*.json`` and flatten to a dict.

    Args:
        datasetsdirpath: Root directory containing one subdirectory per
            dataset.
        valid_datasets: Names of the dataset subdirectories to search for
            ``valid*.json`` files.
        read_type: ``"text"`` maps each ``"{type}_{uid}"`` key to ``""``;
            ``"dict"`` maps it to the full metadata dict.

    Returns:
        Mapping from ``"{type}_{uid}"`` to the value selected by
        ``read_type``.
    """
    if isinstance(datasetsdirpath, str):
        datasetsdirpath = Path(datasetsdirpath)

    valid_json_paths = []
    for valid_dataset in valid_datasets:
        valid_json_paths.extend((datasetsdirpath / valid_dataset).glob("valid*.json"))

    data: Dict[str, str] = {}
    for valid_json_path in valid_json_paths:
        with open(valid_json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        for utt_data in metadata:
            data_type_uid = f"{utt_data['type']}_{utt_data['Uid']}"
            if read_type == "text":
                key, value = data_type_uid, ""
            elif read_type == "dict":
                key, value = data_type_uid, utt_data
            if data_type_uid in data:
                raise RuntimeError(
                    f"{data_type_uid} is duplicated "
                    f"({valid_json_path}:{utt_data['Uid']})"
                )

            data[data_type_uid] = value
    return data


def read_msvs_json_test(
    datasetsdirpath: Union[Path, str],
    test_data: Union[list, Path, str],
    read_type: str = "text",
) -> Dict[str, str]:
    """Read one or more ``test.json`` files and flatten to a dict.

    Args:
        datasetsdirpath: Root directory containing one subdirectory per
            dataset.
        test_data: Either a list of dataset names (each contributing
            ``{dataset}/test.json`` if it exists), or a direct path to a
            single test json file.
        read_type: ``"text"`` maps each uid to ``""``; ``"dict"`` maps to
            the full metadata dict.

    Returns:
        Mapping from utterance id to the value selected by ``read_type``.
    """
    if isinstance(datasetsdirpath, str):
        datasetsdirpath = Path(datasetsdirpath)

    test_json_paths = []
    if isinstance(test_data, list):
        for test_dataset in test_data:
            test_json_path = datasetsdirpath / test_dataset / "test.json"
            if test_json_path.exists():
                test_json_paths.append(test_json_path)
    else:
        if isinstance(test_data, str):
            test_data = Path(test_data)
        test_json_paths.append(test_data)

    data: Dict[str, str] = {}
    for test_json_path in test_json_paths:
        with open(test_json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        for utt_data in metadata:
            uid = utt_data["Uid"]
            if read_type == "text":
                key, value = uid, ""
            elif read_type == "dict":
                key, value = uid, utt_data
            if uid in data:
                raise RuntimeError(
                    f"{uid} is duplicated ({test_json_path}:{utt_data['Uid']})"
                )

            data[uid] = value

    return data


class NpyMSVSJsonReader(collections.abc.Mapping):
    """Reader class for a scp file of numpy file.

    Examples:
        "\n\n"
        "   [    "
        "   {'Uid': 'utterance_id_A', ...},"
        "   {'Uid': 'utterance_id_B', ...},"
        "...]"
        >>> reader = NpyMSVSJsonReader('npy.json')
    """

    def __init__(self, fname: Union[Path, str]):
        assert check_argument_types()
        self.fname = Path(fname)
        self.data = read_msvs_json(fname, read_type="npy")

    def get_path(self, key: str) -> str:
        return self.data[key]

    def __getitem__(self, key: str) -> np.ndarray:
        npy_path = self.data[key]
        return np.load(npy_path)

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()
