#!/usr/bin/env python3
"""CLI that launches a *_train.py-style command under the appropriate
distributed-training wrapper.

Given a Kaldi-style ``cmd`` script (``run.pl``/``queue.pl``/``slurm.pl``) and
a training command (``args``), this figures out how the job should actually
be submitted:

* ``--host`` given: submit one process per (host, GPU id) pair over SSH.
* single node (``--num_nodes <= 1``, no ``--host``): run the training command
  directly through ``cmd``, letting PyTorch handle multi-GPU on this node.
* multiple nodes via ``slurm.pl``: wrap the command in ``srun``.
* multiple nodes via any other queue script: wrap the command in ``mpirun``.

In every multi-process case it also appends the ``--dist_*`` flags the
training script needs to initialize `torch.distributed` (rank, world size,
and an init method — either a shared-file URL or an explicit master
host:port).
"""
import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from espnet2.utils.types import str2bool, str_or_none
from espnet.utils.cli_utils import get_commandline_args


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch distributed process with appropriate options. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # cmd / logging related
    parser.add_argument(
        "--cmd",
        help="The path of cmd script of Kaldi: run.pl. queue.pl, or slurm.pl",
        default="utils/run.pl",
    )
    parser.add_argument(
        "--log",
        help="The path of log file used by cmd",
        default="run.log",
    )
    parser.add_argument(
        "--max_num_log_files",
        help="The maximum number of log-files to be kept",
        default=1000,
    )

    # distributed-training related
    parser.add_argument(
        "--ngpu", type=int, default=1, help="The number of GPUs per node"
    )
    egroup = parser.add_mutually_exclusive_group()
    egroup.add_argument(
        "--num_nodes", type=int, default=1, help="The number of nodes"
    )
    egroup.add_argument(
        "--host",
        type=str,
        default=None,
        help="Directly specify the host names.  The job are submitted via SSH. "
        "Multiple host names can be specified by splitting by comma. e.g. host1,host2"
        " You can also the device id after the host name with ':'. e.g. "
        "host1:0:2:3,host2:0:2. If the device ids are specified in this way, "
        "the value of --ngpu is ignored.",
    )
    parser.add_argument(
        "--envfile",
        type=str_or_none,
        default="path.sh",
        help="Source the shell script before executing command. "
        "This option is used when --host is specified.",
    )
    parser.add_argument(
        "--multiprocessing_distributed",
        type=str2bool,
        default=True,
        help="Distributed method is used when single-node mode.",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=None,
        help="Specify the port number of master"
        "Master is a host machine has RANK0 process.",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default=None,
        help="Specify the address s of master. "
        "Master is a host machine has RANK0 process.",
    )
    parser.add_argument(
        "--init_file_prefix",
        type=str,
        default=".dist_init_",
        help="The file name prefix for init_file, which is used for "
        "'Shared-file system initialization'. "
        "This option is used when --port is not specified",
    )
    parser.add_argument("args", type=str, nargs="+")
    return parser


def _validate_cmd(args: argparse.Namespace) -> None:
    """Fail fast if --cmd doesn't point to a runnable script (SSH mode is exempt)."""
    if args.host is None and shutil.which(args.cmd[0]) is None:
        raise RuntimeError(
            f"The first args of --cmd should be a script path. e.g. utils/run.pl: "
            f"{args.cmd[0]}"
        )


def _resolve_init_method(args: argparse.Namespace) -> Optional[List[str]]:
    """Build the ``--dist_init_method``/``--dist_master_*`` flags for torch.distributed.

    See: https://pytorch.org/docs/stable/distributed.html#initialization
    """
    if args.host is None and args.num_nodes <= 1:
        # Automatically set init_method if num_node=1
        return None

    if args.master_port is None:
        # Try "shared-file system initialization" if master_port is not specified
        # Give random name to avoid reusing previous file
        init_file = Path(args.init_file_prefix + str(uuid.uuid4())).absolute()
        init_file.parent.mkdir(exist_ok=True, parents=True)
        return ["--dist_init_method", f"file://{init_file}"]

    init_method = ["--dist_master_port", str(args.master_port)]
    # This can be omitted if slurm mode
    if args.master_addr is not None:
        init_method += ["--dist_master_addr", args.master_addr]
    elif args.host is not None:
        init_method += [
            "--dist_master_addr",
            args.host.split(",")[0].split(":")[0],
        ]
    return init_method


def _rotate_log_files(log: str, max_num_log_files: int) -> None:
    """Shift run.log -> run.log.1 -> run.log.2 -> ... , dropping the oldest."""
    for i in range(max_num_log_files - 1, -1, -1):
        if i == 0:
            p = Path(log)
            pn = p.parent / (p.stem + ".1" + p.suffix)
        else:
            _p = Path(log)
            p = _p.parent / (_p.stem + f".{i}" + _p.suffix)
            pn = _p.parent / (_p.stem + f".{i + 1}" + _p.suffix)

        if p.exists():
            if i == max_num_log_files - 1:
                p.unlink()
            else:
                shutil.move(p, pn)


def _submit_via_ssh(
    args: argparse.Namespace, init_method: Optional[List[str]]
) -> Tuple[List[subprocess.Popen], List[str]]:
    """Submit one process per (host, GPU id) pair over SSH.

    e.g. ``--host host1:0:2,host2:0:1`` runs 3 processes total: GPUs 0 and 2
    on host1, GPU 0 on host2.
    """
    hosts = []
    ids_list = []
    for host in args.host.split(","):
        # e.g host = "host1:0:2"
        sps = host.split(":")
        host = sps[0]
        if len(sps) > 1:
            ids = [int(x) for x in sps[1:]]
        else:
            ids = list(range(args.ngpu))
        hosts.append(host)
        ids_list.append(ids)

    world_size = sum(max(len(x), 1) for x in ids_list)
    logging.info(f"{len(hosts)}nodes with world_size={world_size} via SSH")

    env = f"source {args.envfile}" if args.envfile is not None else ""

    if args.log != "-":
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        f = Path(args.log).open("w", encoding="utf-8")
    else:
        # Output to stdout/stderr
        f = None

    processes = []
    cmd: List[str] = []
    rank = 0
    for host, ids in zip(hosts, ids_list):
        ngpu = 1 if len(ids) > 0 else 0
        ids = ids if len(ids) > 0 else ["none"]

        for local_rank in ids:
            cmd = (
                args.args
                + [
                    "--ngpu",
                    str(ngpu),
                    "--multiprocessing_distributed",
                    "false",
                    "--local_rank",
                    str(local_rank),
                    "--dist_rank",
                    str(rank),
                    "--dist_world_size",
                    str(world_size),
                ]
                + init_method
            )
            if ngpu == 0:
                # Gloo supports both GPU and CPU mode.
                #   See: https://pytorch.org/docs/stable/distributed.html
                cmd += ["--dist_backend", "gloo"]

            heredoc = f"""<< EOF
set -euo pipefail
cd {os.getcwd()}
{env}
{" ".join([c if len(c) != 0 else "''" for c in cmd])}
EOF
"""

            # FIXME(kamo): The process will be alive
            #  even if this program is stopped because we don't set -t here,
            #  i.e. not assigning pty,
            #  and the program is not killed when SSH connection is closed.
            process = subprocess.Popen(
                ["ssh", host, "bash", heredoc],
                stdout=f,
                stderr=f,
            )
            processes.append(process)
            rank += 1

    return processes, cmd


def _submit_single_node(
    args: argparse.Namespace,
) -> Tuple[List[subprocess.Popen], List[str]]:
    """Run the training command directly on this node (PyTorch handles multi-GPU)."""
    if args.ngpu > 1:
        if args.multiprocessing_distributed:
            # NOTE:
            #   If multiprocessing_distributed=true,
            # -> Distributed mode, which is multi-process and Multi-GPUs.
            #    and TCP initializetion is used if single-node case:
            #      e.g. init_method="tcp://localhost:20000"
            logging.info(f"single-node with {args.ngpu}gpu on distributed mode")
        else:
            # NOTE:
            #   If multiprocessing_distributed=false
            # -> "DataParallel" mode, which is single-process
            #    and Multi-GPUs with threading.
            # See:
            # https://discuss.pytorch.org/t/why-torch-nn-parallel-distributeddataparallel-runs-faster-than-torch-nn-dataparallel-on-single-machine-with-multi-gpu/32977/2
            logging.info(f"single-node with {args.ngpu}gpu using DataParallel")

    # Using cmd as it is simply
    cmd = (
        args.cmd
        # arguments for ${cmd}
        + ["--gpu", str(args.ngpu), args.log]
        # arguments for *_train.py
        + args.args
        + [
            "--ngpu",
            str(args.ngpu),
            "--multiprocessing_distributed",
            str(args.multiprocessing_distributed),
        ]
    )
    process = subprocess.Popen(cmd)
    return [process], cmd


def _submit_slurm(
    args: argparse.Namespace, init_method: List[str]
) -> Tuple[List[subprocess.Popen], List[str]]:
    """Submit a multi-node job via ``srun`` (used when --cmd is slurm.pl)."""
    logging.info(f"{args.num_nodes}nodes and {args.ngpu}gpu-per-node using srun")
    cmd = (
        args.cmd
        # arguments for ${cmd}
        + [
            "--gpu",
            str(args.ngpu),
            "--num_threads",
            str(max(args.ngpu, 1)),
            "--num_nodes",
            str(args.num_nodes),
            args.log,
            "srun",
            # Inherit all environment variable from parent process
            "--export=ALL",
        ]
        # arguments for *_train.py
        + args.args
        + [
            "--ngpu",
            str(args.ngpu),
            "--multiprocessing_distributed",
            "true",
            "--dist_launcher",
            "slurm",
        ]
        + init_method
    )
    if args.ngpu == 0:
        # Gloo supports both GPU and CPU mode.
        #   See: https://pytorch.org/docs/stable/distributed.html
        cmd += ["--dist_backend", "gloo"]
    process = subprocess.Popen(cmd)
    return [process], cmd


def _submit_mpirun(
    args: argparse.Namespace, init_method: List[str]
) -> Tuple[List[subprocess.Popen], List[str]]:
    """Submit a multi-node job via ``mpirun`` (also works with Slurm via queue.pl)."""
    logging.info(f"{args.num_nodes}nodes and {args.ngpu}gpu-per-node using mpirun")
    cmd = (
        args.cmd
        # arguments for ${cmd}
        + [
            "--gpu",
            str(args.ngpu),
            "--num_threads",
            str(max(args.ngpu, 1)),
            # Make sure scheduler setting, i.e. conf/queue.conf
            # so that --num_nodes requires 1process-per-node
            "--num_nodes",
            str(args.num_nodes),
            args.log,
            "mpirun",
            # -np option can be omitted with Torque/PBS
            "-np",
            str(args.num_nodes),
        ]
        # arguments for *_train.py
        + args.args
        + [
            "--ngpu",
            str(args.ngpu),
            "--multiprocessing_distributed",
            "true",
            "--dist_launcher",
            "mpi",
        ]
        + init_method
    )
    if args.ngpu == 0:
        # Gloo supports both GPU and CPU mode.
        #   See: https://pytorch.org/docs/stable/distributed.html
        cmd += ["--dist_backend", "gloo"]
    process = subprocess.Popen(cmd)
    return [process], cmd


def _submit_processes(
    args: argparse.Namespace, init_method: Optional[List[str]]
) -> Tuple[List[subprocess.Popen], List[str]]:
    """Pick and run the appropriate submission strategy, returning its processes."""
    if args.host is not None:
        return _submit_via_ssh(args, init_method)
    elif args.num_nodes <= 1:
        return _submit_single_node(args)
    elif Path(args.cmd[0]).name == "run.pl":
        raise RuntimeError("run.pl doesn't support submitting to the other nodes.")
    elif Path(args.cmd[0]).name == "ssh.pl":
        raise RuntimeError("Use --host option instead of ssh.pl")
    elif Path(args.cmd[0]).name == "slurm.pl":
        return _submit_slurm(args, init_method)
    else:
        # This pattern can also works with Slurm.
        return _submit_mpirun(args, init_method)


def _wait_for_processes(
    processes: List[subprocess.Popen], cmd: List[str], log: str
) -> None:
    """Block until every process exits, killing the rest as soon as one fails."""
    failed = False
    while any(p.returncode is None for p in processes):
        for process in processes:
            # If any process is failed, try to kill the other processes too
            if failed and process.returncode is not None:
                process.kill()
            else:
                try:
                    process.wait(0.5)
                except subprocess.TimeoutExpired:
                    pass

                if process.returncode is not None and process.returncode != 0:
                    failed = True

    for process in processes:
        if process.returncode != 0:
            print(
                subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd),
                file=sys.stderr,
            )
            p = Path(log)
            if p.exists():
                with p.open() as f:
                    lines = list(f)
                raise RuntimeError(
                    f"\n################### The last 1000 lines of {log} "
                    f"###################\n" + "".join(lines[-1000:])
                )
            else:
                raise RuntimeError


def main(cmd: Optional[List[str]] = None) -> None:
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = get_parser()
    args = parser.parse_args(cmd)
    args.cmd = shlex.split(args.cmd)

    _validate_cmd(args)
    init_method = _resolve_init_method(args)
    _rotate_log_files(args.log, args.max_num_log_files)

    processes, submitted_cmd = _submit_processes(args, init_method)
    logging.info(f"log file: {args.log}")
    _wait_for_processes(processes, submitted_cmd, args.log)


if __name__ == "__main__":
    main()
