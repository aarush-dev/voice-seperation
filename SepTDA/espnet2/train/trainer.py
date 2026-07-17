"""Trainer module.

This module implements :class:`Trainer`, which drives the full training
loop shared by (almost) every ESPnet2 recipe:

* build/resume optimizer, scheduler, and AMP-scaler state,
* wrap the model for (distributed) data parallelism,
* for each epoch: run one training epoch, one validation epoch, and
  (optionally) attention plotting,
* step epoch-level LR schedulers,
* log to the console / matplotlib / tensorboard / wandb via
  :class:`espnet2.train.reporter.Reporter`,
* checkpoint the run, track the n-best models, and average them,
* stop early on a plateau or when no valid gradient step occurred.

The loop is written as a set of ``classmethod``\\ s precisely so that
subclasses (e.g. :class:`espnet2.train.gan_trainer.GANTrainer` and
:class:`espnet2.train.uasr_trainer.UASRTrainer`) can override only the
per-batch logic (``train_one_epoch`` / ``validate_one_epoch``) while
reusing all of the surrounding checkpointing/reporting machinery in
``run()``.
"""
import argparse
import dataclasses
import logging
import time
from contextlib import contextmanager
from dataclasses import is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import humanfriendly
import numpy as np
import torch
import torch.nn
import torch.optim
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.main_funcs.average_nbest_models import average_nbest_models
from espnet2.main_funcs.calculate_all_attentions import calculate_all_attentions
from espnet2.schedulers.abs_scheduler import (
    AbsBatchStepScheduler,
    AbsEpochStepScheduler,
    AbsScheduler,
    AbsValEpochStepScheduler,
)
from espnet2.torch_utils.add_gradient_noise import add_gradient_noise
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import Reporter, SubReporter
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils.kwargs2args import kwargs2args

if torch.distributed.is_available():
    from torch.distributed import ReduceOp

autocast_args = dict()
if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import GradScaler, autocast

    if (
        V(torch.__version__) >= V("1.10.0")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        autocast_args = dict(dtype=torch.float16)
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

    GradScaler = None

try:
    import fairscale
except ImportError:
    fairscale = None


@dataclasses.dataclass
class TrainerOptions:
    """Configuration consumed by :class:`Trainer`'s classmethods.

    An instance is built by :meth:`Trainer.build_options` from the parsed
    command-line namespace (see ``espnet2.utils.build_dataclass``); each
    field mirrors a ``--xxx`` CLI option defined in ``espnet2.tasks.abs_task``.
    """

    ngpu: int
    resume: bool
    use_amp: bool
    train_dtype: str
    grad_noise: bool
    accum_grad: int
    grad_clip: float
    grad_clip_type: float
    log_interval: Optional[int]
    no_forward_run: bool
    use_matplotlib: bool
    use_tensorboard: bool
    use_wandb: bool
    output_dir: Union[Path, str]
    max_epoch: int
    seed: int
    sharded_ddp: bool
    patience: Optional[int]
    keep_nbest_models: Union[int, List[int]]
    nbest_averaging_interval: int
    early_stopping_criterion: Sequence[str]
    best_model_criterion: Sequence[Sequence[str]]
    val_scheduler_criterion: Sequence[str]
    unused_parameters: bool
    wandb_model_log_interval: int
    create_graph_in_tensorboard: bool


class Trainer:
    """Drives training given a model, optimizer(s), and data iterators.

    If you'd like to use multiple optimizers, then inherit this class
    and override the methods if necessary - at least "train_one_epoch()"

    >>> class TwoOptimizerTrainer(Trainer):
    ...     @classmethod
    ...     def add_arguments(cls, parser):
    ...         ...
    ...
    ...     @classmethod
    ...     def train_one_epoch(cls, model, optimizers, ...):
    ...         loss1 = model.model1(...)
    ...         loss1.backward()
    ...         optimizers[0].step()
    ...
    ...         loss2 = model.model2(...)
    ...         loss2.backward()
    ...         optimizers[1].step()

    """

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    def build_options(cls, args: argparse.Namespace) -> TrainerOptions:
        """Build options consumed by train(), eval(), and plot_attention()"""
        assert check_argument_types()
        return build_dataclass(TrainerOptions, args)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Reserved for future development of another Trainer"""
        pass

    @staticmethod
    def resume(
        checkpoint: Union[str, Path],
        model: torch.nn.Module,
        reporter: Reporter,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional["GradScaler"],
        ngpu: int = 0,
    ) -> None:
        """Restore model/optimizer/scheduler/reporter/scaler state in-place.

        Args:
            checkpoint: Path to a ``checkpoint.pth`` written by
                :meth:`Trainer._save_checkpoint`.
            model: Model whose ``state_dict`` is loaded (non-strict, so
                extra/missing keys are tolerated).
            reporter: Reporter to restore epoch/stats history into.
            optimizers: Optimizers to restore, matched by position to the
                saved list.
            schedulers: Schedulers to restore, matched by position
                (``None`` entries are skipped).
            scaler: AMP gradient scaler to restore, if AMP is enabled.
            ngpu: Number of GPUs; used to pick the map-location for loading.
        """
        states = torch.load(
            checkpoint,
            map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu",
            weights_only=False,
        )
        model.load_state_dict(states["model"], strict=False)
        reporter.load_state_dict(states["reporter"])
        for optimizer, state in zip(optimizers, states["optimizers"]):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, states["schedulers"]):
            if scheduler is not None:
                scheduler.load_state_dict(state)
        if scaler is not None:
            if states["scaler"] is None:
                logging.warning("scaler state is not found")
            else:
                scaler.load_state_dict(states["scaler"])

        logging.info(f"The training was resumed using {checkpoint}")

    # ------------------------------------------------------------------
    # run() and its helpers
    # ------------------------------------------------------------------

    @classmethod
    def run(
        cls,
        model: AbsESPnetModel,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        train_iter_factory: AbsIterFactory,
        valid_iter_factory: AbsIterFactory,
        plot_attention_iter_factory: Optional[AbsIterFactory],
        trainer_options,
        distributed_option: DistributedOption,
    ) -> None:
        """Perform training. This method performs the main process of training.

        Runs from ``reporter.get_epoch() + 1`` (0 on a fresh run, or the
        resumed epoch) through ``trainer_options.max_epoch``, calling
        :meth:`train_one_epoch` / :meth:`validate_one_epoch` each epoch and
        handling checkpointing, best-model tracking, n-best pruning, and
        early stopping. Only rank 0 performs I/O in distributed training.
        """
        assert check_argument_types()
        # NOTE(kamo): Don't check the type more strictly as far trainer_options
        assert is_dataclass(trainer_options), type(trainer_options)
        assert len(optimizers) == len(schedulers), (len(optimizers), len(schedulers))

        keep_nbest_models = cls._resolve_keep_nbest_models(trainer_options)

        output_dir = Path(trainer_options.output_dir)
        reporter = Reporter()
        scaler = cls._build_grad_scaler(trainer_options)

        if trainer_options.resume and (output_dir / "checkpoint.pth").exists():
            cls.resume(
                checkpoint=output_dir / "checkpoint.pth",
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                reporter=reporter,
                scaler=scaler,
                ngpu=trainer_options.ngpu,
            )

        start_epoch = reporter.get_epoch() + 1
        if start_epoch == trainer_options.max_epoch + 1:
            logging.warning(
                f"The training has already reached at max_epoch: {start_epoch}"
            )

        dp_model = cls._wrap_model_for_training(
            model, optimizers, trainer_options, distributed_option
        )
        train_summary_writer, valid_summary_writer = cls._build_tensorboard_writers(
            trainer_options, distributed_option, output_dir
        )

        is_rank0 = (
            not distributed_option.distributed or distributed_option.dist_rank == 0
        )

        start_time = time.perf_counter()
        for iepoch in range(start_epoch, trainer_options.max_epoch + 1):
            cls._log_epoch_start(iepoch, start_epoch, trainer_options, start_time)
            set_all_random_seed(trainer_options.seed + iepoch)

            reporter.set_epoch(iepoch)
            # 1. Train and validation for one-epoch
            with reporter.observe("train") as sub_reporter:
                all_steps_are_invalid = cls.train_one_epoch(
                    model=dp_model,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    iterator=train_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    scaler=scaler,
                    summary_writer=train_summary_writer,
                    options=trainer_options,
                    distributed_option=distributed_option,
                )

            with reporter.observe("valid") as sub_reporter:
                cls.validate_one_epoch(
                    model=dp_model,
                    iterator=valid_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    options=trainer_options,
                    distributed_option=distributed_option,
                )
            if is_rank0:
                # att_plot doesn't support distributed
                if plot_attention_iter_factory is not None:
                    with reporter.observe("att_plot") as sub_reporter:
                        cls.plot_attention(
                            model=model,
                            output_dir=output_dir / "att_ws",
                            summary_writer=train_summary_writer,
                            iterator=plot_attention_iter_factory.build_iter(iepoch),
                            reporter=sub_reporter,
                            options=trainer_options,
                        )

            # 2. LR Scheduler step
            cls._step_epoch_schedulers(schedulers, reporter, trainer_options)
            cls._consolidate_sharded_optimizer_states(optimizers, trainer_options)

            if is_rank0:
                # 3. Report the results
                cls._log_and_plot_epoch_results(
                    reporter,
                    trainer_options,
                    output_dir,
                    train_summary_writer,
                    valid_summary_writer,
                )

                # 4. Save/Update the checkpoint
                cls._save_checkpoint(
                    output_dir, model, reporter, optimizers, schedulers, scaler
                )

                # 5. Save and log the model and update the link to the best model
                cls._save_epoch_model_and_update_latest_link(
                    output_dir, model, iepoch
                )
                cls._update_best_model_links(
                    reporter, output_dir, iepoch, trainer_options
                )

                # 6. Remove the model files excluding n-best epoch and latest epoch
                cls._remove_stale_epoch_checkpoints(
                    output_dir, reporter, trainer_options, keep_nbest_models, iepoch
                )

            # 7. If any updating haven't happened, stops the training
            if all_steps_are_invalid:
                logging.warning(
                    "The gradients at all steps are invalid in this epoch. "
                    f"Something seems wrong. This training was stopped at {iepoch}epoch"
                )
                break

            # 8. Check early stopping
            if trainer_options.patience is not None:
                if reporter.check_early_stopping(
                    trainer_options.patience, *trainer_options.early_stopping_criterion
                ):
                    break

        else:
            logging.info(
                f"The training was finished at {trainer_options.max_epoch} epochs "
            )

        # Generated n-best averaged model
        if is_rank0:
            average_nbest_models(
                reporter=reporter,
                output_dir=output_dir,
                best_model_criterion=trainer_options.best_model_criterion,
                nbest=keep_nbest_models,
            )

    @staticmethod
    def _resolve_keep_nbest_models(options: TrainerOptions) -> List[int]:
        """Normalize ``options.keep_nbest_models`` into a non-empty list of ints."""
        if isinstance(options.keep_nbest_models, int):
            return [options.keep_nbest_models]
        if len(options.keep_nbest_models) == 0:
            logging.warning("No keep_nbest_models is given. Change to [1]")
            options.keep_nbest_models = [1]
        return options.keep_nbest_models

    @staticmethod
    def _build_grad_scaler(options: TrainerOptions) -> Optional["GradScaler"]:
        """Create the AMP gradient scaler, or ``None`` if AMP is disabled."""
        if not options.use_amp:
            return None
        if V(torch.__version__) < V("1.6.0"):
            raise RuntimeError("Require torch>=1.6.0 for  Automatic Mixed Precision")
        if options.sharded_ddp:
            if fairscale is None:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            return fairscale.optim.grad_scaler.ShardedGradScaler()
        return GradScaler(init_scale=1280.0)

    @staticmethod
    def _wrap_model_for_training(
        model: AbsESPnetModel,
        optimizers: Sequence[torch.optim.Optimizer],
        options: TrainerOptions,
        distributed_option: DistributedOption,
    ) -> torch.nn.Module:
        """Wrap ``model`` with (Sharded)DistributedDataParallel/DataParallel."""
        if distributed_option.distributed:
            if options.sharded_ddp:
                return fairscale.nn.data_parallel.ShardedDataParallel(
                    module=model,
                    sharded_optimizer=optimizers,
                )
            return torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=(
                    # Perform multi-Process with multi-GPUs
                    [torch.cuda.current_device()]
                    if distributed_option.ngpu == 1
                    # Perform single-Process with multi-GPUs
                    else None
                ),
                output_device=(
                    torch.cuda.current_device()
                    if distributed_option.ngpu == 1
                    else None
                ),
                find_unused_parameters=options.unused_parameters,
            )
        elif distributed_option.ngpu > 1:
            return torch.nn.parallel.DataParallel(
                model,
                device_ids=list(range(distributed_option.ngpu)),
            )
        else:
            # NOTE(kamo): DataParallel also should work with ngpu=1,
            # but for debuggability it's better to keep this block.
            return model

    @staticmethod
    def _build_tensorboard_writers(
        options: TrainerOptions,
        distributed_option: DistributedOption,
        output_dir: Path,
    ) -> Tuple[Optional["object"], Optional["object"]]:
        """Create (train, valid) ``SummaryWriter``\\ s, or ``(None, None)``."""
        if options.use_tensorboard and (
            not distributed_option.distributed or distributed_option.dist_rank == 0
        ):
            from torch.utils.tensorboard import SummaryWriter

            train_summary_writer = SummaryWriter(
                str(output_dir / "tensorboard" / "train")
            )
            valid_summary_writer = SummaryWriter(
                str(output_dir / "tensorboard" / "valid")
            )
            return train_summary_writer, valid_summary_writer
        return None, None

    @staticmethod
    def _log_epoch_start(
        iepoch: int, start_epoch: int, options: TrainerOptions, start_time: float
    ) -> None:
        """Log the epoch-start message, with an ETA once one epoch has run."""
        if iepoch != start_epoch:
            logging.info(
                "{}/{}epoch started. Estimated time to finish: {}".format(
                    iepoch,
                    options.max_epoch,
                    humanfriendly.format_timespan(
                        (time.perf_counter() - start_time)
                        / (iepoch - start_epoch)
                        * (options.max_epoch - iepoch + 1)
                    ),
                )
            )
        else:
            logging.info(f"{iepoch}/{options.max_epoch}epoch started")

    @staticmethod
    def _step_epoch_schedulers(
        schedulers: Sequence[Optional[AbsScheduler]],
        reporter: Reporter,
        options: TrainerOptions,
    ) -> None:
        """Step every epoch-level (as opposed to per-batch) LR scheduler."""
        for scheduler in schedulers:
            if isinstance(scheduler, AbsValEpochStepScheduler):
                scheduler.step(reporter.get_value(*options.val_scheduler_criterion))
            elif isinstance(scheduler, AbsEpochStepScheduler):
                scheduler.step()

    @staticmethod
    def _consolidate_sharded_optimizer_states(
        optimizers: Sequence[torch.optim.Optimizer], options: TrainerOptions
    ) -> None:
        """Gather sharded (OSS) optimizer state onto rank 0 before saving."""
        if options.sharded_ddp:
            for optimizer in optimizers:
                if isinstance(optimizer, fairscale.optim.oss.OSS):
                    optimizer.consolidate_state_dict()

    @staticmethod
    def _log_and_plot_epoch_results(
        reporter: Reporter,
        options: TrainerOptions,
        output_dir: Path,
        train_summary_writer,
        valid_summary_writer,
    ) -> None:
        """Log the epoch summary and forward it to matplotlib/tensorboard/wandb."""
        logging.info(reporter.log_message())
        if options.use_matplotlib:
            reporter.matplotlib_plot(output_dir / "images")
        if train_summary_writer is not None:
            reporter.tensorboard_add_scalar(train_summary_writer, key1="train")
            reporter.tensorboard_add_scalar(valid_summary_writer, key1="valid")
        if options.use_wandb:
            reporter.wandb_log()

    @staticmethod
    def _save_checkpoint(
        output_dir: Path,
        model: torch.nn.Module,
        reporter: Reporter,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional["GradScaler"],
    ) -> None:
        """Write the resumable ``checkpoint.pth`` (model + training state)."""
        torch.save(
            {
                "model": model.state_dict(),
                "reporter": reporter.state_dict(),
                "optimizers": [o.state_dict() for o in optimizers],
                "schedulers": [
                    s.state_dict() if s is not None else None for s in schedulers
                ],
                "scaler": scaler.state_dict() if scaler is not None else None,
            },
            output_dir / "checkpoint.pth",
        )

    @staticmethod
    def _save_epoch_model_and_update_latest_link(
        output_dir: Path, model: torch.nn.Module, iepoch: int
    ) -> None:
        """Save ``{iepoch}epoch.pth`` and repoint ``latest.pth`` to it."""
        torch.save(model.state_dict(), output_dir / f"{iepoch}epoch.pth")

        # Creates a sym link latest.pth -> {iepoch}epoch.pth
        latest_path = output_dir / "latest.pth"
        if latest_path.is_symlink() or latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(f"{iepoch}epoch.pth")

    @classmethod
    def _update_best_model_links(
        cls,
        reporter: Reporter,
        output_dir: Path,
        iepoch: int,
        options: TrainerOptions,
    ) -> List[str]:
        """Symlink ``{phase}.{metric}.best.pth`` for criteria improved this epoch.

        Also logs the freshly-saved epoch model as a W&B model artifact when
        ``options.wandb_model_log_interval`` says this epoch is due.

        Returns:
            The list of ``"{phase}.{metric}"`` strings that improved this epoch.
        """
        improved = []
        for _phase, k, _mode in options.best_model_criterion:
            # e.g. _phase, k, _mode = "train", "loss", "min"
            if reporter.has(_phase, k):
                best_epoch = reporter.get_best_epoch(_phase, k, _mode)
                # Creates sym links if it's the best result
                if best_epoch == iepoch:
                    p = output_dir / f"{_phase}.{k}.best.pth"
                    if p.is_symlink() or p.exists():
                        p.unlink()
                    p.symlink_to(f"{iepoch}epoch.pth")
                    improved.append(f"{_phase}.{k}")
        if len(improved) == 0:
            logging.info("There are no improvements in this epoch")
        else:
            logging.info("The best model has been updated: " + ", ".join(improved))

        log_model = (
            options.wandb_model_log_interval > 0
            and iepoch % options.wandb_model_log_interval == 0
        )
        if log_model and options.use_wandb:
            import wandb

            logging.info("Logging Model on this epoch :::::")
            artifact = wandb.Artifact(
                name=f"model_{wandb.run.id}",
                type="model",
                metadata={"improved": improved},
            )
            artifact.add_file(str(output_dir / f"{iepoch}epoch.pth"))
            aliases = [
                f"epoch-{iepoch}",
                "best" if best_epoch == iepoch else "",
            ]
            wandb.log_artifact(artifact, aliases=aliases)

        return improved

    @staticmethod
    def _remove_stale_epoch_checkpoints(
        output_dir: Path,
        reporter: Reporter,
        options: TrainerOptions,
        keep_nbest_models: List[int],
        iepoch: int,
    ) -> None:
        """Delete ``{e}epoch.pth`` files outside the union of n-best epochs.

        Also triggers periodic n-best-model averaging when
        ``options.nbest_averaging_interval`` divides ``iepoch``.
        """
        removed = []
        # Get the union set of the n-best among multiple criterion
        nbests = set().union(
            *[
                set(reporter.sort_epochs(ph, k, m)[: max(keep_nbest_models)])
                for ph, k, m in options.best_model_criterion
                if reporter.has(ph, k)
            ]
        )

        # Generated n-best averaged model
        if (
            options.nbest_averaging_interval > 0
            and iepoch % options.nbest_averaging_interval == 0
        ):
            average_nbest_models(
                reporter=reporter,
                output_dir=output_dir,
                best_model_criterion=options.best_model_criterion,
                nbest=keep_nbest_models,
                suffix=f"till{iepoch}epoch",
            )

        for e in range(1, iepoch):
            p = output_dir / f"{e}epoch.pth"
            if p.exists() and e not in nbests:
                p.unlink()
                removed.append(str(p))
        if len(removed) != 0:
            logging.info("The model files were removed: " + ", ".join(removed))

    # ------------------------------------------------------------------
    # train_one_epoch() and its helpers
    # ------------------------------------------------------------------

    @classmethod
    def train_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional["GradScaler"],
        reporter: SubReporter,
        summary_writer,
        options: TrainerOptions,
        distributed_option: DistributedOption,
    ) -> bool:
        """Run one training epoch: forward/backward/optimizer-step per batch.

        Gradients are accumulated over ``options.accum_grad`` minibatches;
        every ``accum_grad``-th step, gradients are (optionally noised,
        then) clipped and, if finite, applied via the optimizer(s) and any
        per-batch LR scheduler. Stats are pushed into ``reporter`` and
        periodically logged / sent to tensorboard / wandb.

        Returns:
            True if every accumulation step in this epoch produced a
            non-finite gradient norm, i.e. no optimizer step ever happened.
        """
        assert check_argument_types()

        grad_noise = options.grad_noise
        accum_grad = options.accum_grad
        grad_clip = options.grad_clip
        grad_clip_type = options.grad_clip_type
        log_interval = cls._resolve_log_interval(iterator, options.log_interval)
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        use_wandb = options.use_wandb
        create_graph_in_tensorboard = options.create_graph_in_tensorboard
        distributed = distributed_option.distributed

        model.train()
        all_steps_are_invalid = True
        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        start_time = time.perf_counter()
        for iiter, (utt_id, batch) in enumerate(
            reporter.measure_iter_time(iterator, "iter_time"), 1
        ):
            assert isinstance(batch, dict), type(batch)

            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch["utt_id"] = utt_id

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                all_steps_are_invalid = False
                continue

            if (
                create_graph_in_tensorboard
                and iiter == 1
                and summary_writer is not None
            ):
                cls._add_graph_to_tensorboard(model, batch, summary_writer, distributed)

            with autocast(
                scaler is not None,
                **autocast_args,
            ):
                with reporter.measure_time("forward_time"):
                    loss, stats, weight, optim_idx = cls._forward_and_collect_loss(
                        model, batch
                    )

                stats = {k: v for k, v in stats.items() if v is not None}
                if ngpu > 1 or distributed:
                    # Apply weighted averaging for loss and stats
                    loss = (loss * weight.type(loss.dtype)).sum()

                    # if distributed, this method can also apply all_reduce()
                    stats, weight = recursive_average(stats, weight, distributed)

                    # Now weight is summation over all workers
                    loss /= weight
                if distributed:
                    # NOTE(kamo): Multiply world_size because DistributedDataParallel
                    # automatically normalizes the gradient by world_size.
                    loss *= torch.distributed.get_world_size()

                loss /= accum_grad

            reporter.register(stats, weight)

            with reporter.measure_time("backward_time"):
                if scaler is not None:
                    # Scales loss.  Calls backward() on scaled loss
                    # to create scaled gradients.
                    # Backward passes under autocast are not recommended.
                    # Backward ops run in the same dtype autocast chose
                    # for corresponding forward ops.
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if iiter % accum_grad == 0:
                grad_was_valid, start_time = cls._optimizer_step(
                    model=model,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    scaler=scaler,
                    optim_idx=optim_idx,
                    grad_clip=grad_clip,
                    grad_clip_type=grad_clip_type,
                    grad_noise=grad_noise,
                    reporter=reporter,
                    start_time=start_time,
                )
                if grad_was_valid:
                    all_steps_are_invalid = False

            # NOTE(kamo): Call log_message() after next()
            reporter.next()
            if iiter % log_interval == 0:
                logging.info(reporter.log_message(-log_interval))
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer, -log_interval)
                if use_wandb:
                    reporter.wandb_log()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
        return all_steps_are_invalid

    @staticmethod
    def _resolve_log_interval(iterator, log_interval: Optional[int]) -> int:
        """Fall back to ~20 log lines per epoch (min 10) when unspecified."""
        if log_interval is not None:
            return log_interval
        try:
            return max(len(iterator) // 20, 10)
        except TypeError:
            return 100

    @staticmethod
    def _add_graph_to_tensorboard(
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        summary_writer,
        distributed: bool,
    ) -> None:
        """Best-effort attempt to log the model graph to tensorboard.

        Only called once, on the first minibatch, when
        ``create_graph_in_tensorboard`` is enabled.
        """
        if distributed:
            _model = getattr(model, "module")
        else:
            _model = model
            if _model is not None:
                try:
                    _args = kwargs2args(_model.forward, batch)
                except (ValueError, TypeError):
                    logging.warning(
                        "inpect.signature() is failed for the model. "
                        "The graph can't be added for tensorboard."
                    )
                else:
                    try:
                        summary_writer.add_graph(_model, _args, use_strict_trace=False)
                    except Exception:
                        logging.warning(
                            "summary_writer.add_graph() "
                            "is failed for the model. "
                            "The graph can't be added for tensorboard."
                        )
                    del _args
            else:
                logging.warning("model.module is not found (This should be a bug.)")
        del _model

    @staticmethod
    def _normalize_optim_idx(optim_idx):
        """Coerce a model-reported ``optim_idx`` to a plain int (or None).

        ``optim_idx`` may be an int already, or a 0/1-dim ``torch.Tensor``;
        in the 1-dim case every entry must agree on the same value (this
        happens when the value is broadcast across a batch dimension).
        """
        if optim_idx is not None and not isinstance(optim_idx, int):
            if not isinstance(optim_idx, torch.Tensor):
                raise RuntimeError(
                    "optim_idx must be int or 1dim torch.Tensor, "
                    f"but got {type(optim_idx)}"
                )
            if optim_idx.dim() >= 2:
                raise RuntimeError(
                    "optim_idx must be int or 1dim torch.Tensor, "
                    f"but got {optim_idx.dim()}dim tensor"
                )
            if optim_idx.dim() == 1:
                for v in optim_idx:
                    if v != optim_idx[0]:
                        raise RuntimeError(
                            "optim_idx must be 1dim tensor "
                            "having same values for all entries"
                        )
                optim_idx = optim_idx[0].item()
            else:
                optim_idx = optim_idx.item()
        return optim_idx

    @classmethod
    def _forward_and_collect_loss(
        cls, model: torch.nn.Module, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Optional[int]]:
        """Run the forward pass and normalize the model's return value.

        The model may return either a dict with "loss"/"stats"/"weight"/
        (optional) "optim_idx" keys, or a plain ``(loss, stats, weight)``
        tuple, in which case ``optim_idx`` is ``None``.
        """
        retval = model(**batch)

        # Note(kamo):
        # Supporting two patterns for the returned value from the model
        #   a. dict type
        if isinstance(retval, dict):
            loss = retval["loss"]
            stats = retval["stats"]
            weight = retval["weight"]
            optim_idx = cls._normalize_optim_idx(retval.get("optim_idx"))

        #   b. tuple or list type
        else:
            loss, stats, weight = retval
            optim_idx = None
        return loss, stats, weight, optim_idx

    @staticmethod
    def _optimizer_step(
        model: torch.nn.Module,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional["GradScaler"],
        optim_idx: Optional[int],
        grad_clip: float,
        grad_clip_type: float,
        grad_noise: bool,
        reporter: SubReporter,
        start_time: float,
    ) -> Tuple[bool, float]:
        """Clip gradients and step the optimizer(s)/batch-schedulers once.

        Called every ``accum_grad`` minibatches. If ``optim_idx`` is not
        None, only the optimizer/scheduler at that index is touched
        (multi-optimizer models select which one owns the current loss).

        Returns:
            A ``(grad_was_valid, next_start_time)`` pair: whether the
            gradient norm was finite (so an optimizer step actually
            happened), and the new timer start for the next accumulation
            window.
        """
        if scaler is not None:
            # Unscales the gradients of optimizer's assigned params in-place
            for iopt, optimizer in enumerate(optimizers):
                if optim_idx is not None and iopt != optim_idx:
                    continue
                scaler.unscale_(optimizer)

        # gradient noise injection
        if grad_noise:
            add_gradient_noise(
                model,
                reporter.get_total_count(),
                duration=100,
                eta=1.0,
                scale_factor=0.55,
            )

        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip,
            norm_type=grad_clip_type,
        )
        # PyTorch<=1.4, clip_grad_norm_ returns float value
        if not isinstance(grad_norm, torch.Tensor):
            grad_norm = torch.tensor(grad_norm)

        grad_was_valid = bool(torch.isfinite(grad_norm))
        if not grad_was_valid:
            logging.warning(f"The grad norm is {grad_norm}. Skipping updating the model.")

            # Must invoke scaler.update() if unscale_() is used in the iteration
            # to avoid the following error:
            #   RuntimeError: unscale_() has already been called
            #   on this optimizer since the last update().
            # Note that if the gradient has inf/nan values,
            # scaler.step skips optimizer.step().
            if scaler is not None:
                for iopt, optimizer in enumerate(optimizers):
                    if optim_idx is not None and iopt != optim_idx:
                        continue
                    scaler.step(optimizer)
                    scaler.update()

        else:
            reporter.register(
                {
                    "grad_norm": grad_norm,
                    "clip": torch.where(
                        grad_norm > grad_clip,
                        grad_norm.new_tensor(100),
                        grad_norm.new_tensor(0),
                    ),
                    "loss_scale": scaler.get_scale() if scaler else 1.0,
                }
            )
            with reporter.measure_time("optim_step_time"):
                for iopt, (optimizer, scheduler) in enumerate(
                    zip(optimizers, schedulers)
                ):
                    if optim_idx is not None and iopt != optim_idx:
                        continue
                    if scaler is not None:
                        # scaler.step() first unscales the gradients of
                        # the optimizer's assigned params.
                        scaler.step(optimizer)
                        # Updates the scale for next iteration.
                        scaler.update()
                    else:
                        optimizer.step()
                    if isinstance(scheduler, AbsBatchStepScheduler):
                        scheduler.step()

        for iopt, optimizer in enumerate(optimizers):
            if optim_idx is not None and iopt != optim_idx:
                continue
            optimizer.zero_grad()

        # Register lr and train/load time[sec/step],
        # where step refers to accum_grad * mini-batch
        reporter.register(
            dict(
                {
                    f"optim{i}_lr{j}": pg["lr"]
                    for i, optimizer in enumerate(optimizers)
                    for j, pg in enumerate(optimizer.param_groups)
                    if "lr" in pg
                },
                train_time=time.perf_counter() - start_time,
            ),
        )
        return grad_was_valid, time.perf_counter()

    # ------------------------------------------------------------------
    # validate_one_epoch()
    # ------------------------------------------------------------------

    @classmethod
    @torch.no_grad()
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
        distributed_option: DistributedOption,
    ) -> None:
        """Run one validation epoch (no gradients, no optimizer steps)."""
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = distributed_option.distributed

        model.eval()

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        for utt_id, batch in iterator:
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch["utt_id"] = utt_id

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            stats, weight = cls._forward_and_collect_stats(model, batch)
            if ngpu > 1 or distributed:
                # Apply weighted averaging for stats.
                # if distributed, this method can also apply all_reduce()
                stats, weight = recursive_average(stats, weight, distributed)

            reporter.register(stats, weight)
            reporter.next()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

    @staticmethod
    def _forward_and_collect_stats(
        model: torch.nn.Module, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Run the forward pass and extract (stats, weight) for validation."""
        retval = model(**batch)
        if isinstance(retval, dict):
            stats = retval["stats"]
            weight = retval["weight"]
        else:
            _, stats, weight = retval
        return stats, weight

    # ------------------------------------------------------------------
    # plot_attention() and its helpers
    # ------------------------------------------------------------------

    @classmethod
    @torch.no_grad()
    def plot_attention(
        cls,
        model: torch.nn.Module,
        output_dir: Optional[Path],
        summary_writer,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        reporter: SubReporter,
        options: TrainerOptions,
    ) -> None:
        """Compute and save attention-weight plots for a handful of batches."""
        assert check_argument_types()
        import matplotlib

        ngpu = options.ngpu
        no_forward_run = options.no_forward_run

        matplotlib.use("Agg")

        model.eval()
        for ids, batch in iterator:
            assert isinstance(batch, dict), type(batch)
            assert len(next(iter(batch.values()))) == len(ids), (
                len(next(iter(batch.values()))),
                len(ids),
            )

            batch["utt_id"] = ids

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            # 1. Forwarding model and gathering all attentions
            #    calculate_all_attentions() uses single gpu only.
            att_dict = calculate_all_attentions(model, batch)

            # 2. Plot attentions: This part is slow due to matplotlib
            for k, att_list in att_dict.items():
                assert len(att_list) == len(ids), (len(att_list), len(ids))
                for id_, att_w in zip(ids, att_list):
                    cls._plot_and_save_one_attention(
                        att_w, k, id_, output_dir, summary_writer, reporter, options
                    )
            reporter.next()

    @staticmethod
    def _normalize_attention_weight(att_w) -> np.ndarray:
        """Convert an attention weight to a ``(head, in, out)`` numpy array."""
        if isinstance(att_w, torch.Tensor):
            att_w = att_w.detach().cpu().numpy()

        if att_w.ndim == 2:
            att_w = att_w[None]
        elif att_w.ndim == 4:
            # In multispkr_asr model case, the dimension could be 4.
            att_w = np.concatenate([att_w[i] for i in range(att_w.shape[0])], axis=0)
        elif att_w.ndim > 4 or att_w.ndim == 1:
            raise RuntimeError(f"Must be 2, 3 or 4 dimension: {att_w.ndim}")
        return att_w

    @classmethod
    def _plot_and_save_one_attention(
        cls,
        att_w,
        k: str,
        id_: str,
        output_dir: Optional[Path],
        summary_writer,
        reporter: SubReporter,
        options: TrainerOptions,
    ) -> None:
        """Render one utterance's attention map(s) and save/log the figure."""
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        att_w = cls._normalize_attention_weight(att_w)

        w, h = plt.figaspect(1.0 / len(att_w))
        fig = plt.Figure(figsize=(w * 1.3, h * 1.3))
        axes = fig.subplots(1, len(att_w))
        if len(att_w) == 1:
            axes = [axes]

        for ax, aw in zip(axes, att_w):
            ax.imshow(aw.astype(np.float32), aspect="auto")
            ax.set_title(f"{k}_{id_}")
            ax.set_xlabel("Input")
            ax.set_ylabel("Output")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        if output_dir is not None:
            p = output_dir / id_ / f"{k}.{reporter.get_epoch()}ep.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p)

        if summary_writer is not None:
            summary_writer.add_figure(f"{k}_{id_}", fig, reporter.get_epoch())

        if options.use_wandb:
            import wandb

            wandb.log({f"attention plot/{k}_{id_}": wandb.Image(fig)})
