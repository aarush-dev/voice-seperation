# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Trainer module for GAN-based training.

:class:`GANTrainer` reuses all of :class:`espnet2.train.trainer.Trainer`'s
epoch/checkpoint/reporting machinery (``run()``), but replaces the per-batch
logic with an alternating generator/discriminator update: each minibatch
runs a "generator" turn and a "discriminator" turn (order controlled by
``generator_first``), each with its own forward pass, backward pass, and
optimizer step. Unlike the base :class:`Trainer`, gradient accumulation
(``accum_grad > 1``) and gradient-noise injection are not supported.
"""

import argparse
import dataclasses
import logging
import time
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler, AbsScheduler
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import SubReporter
from espnet2.train.trainer import Trainer, TrainerOptions
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils.types import str2bool

if torch.distributed.is_available():
    from torch.distributed import ReduceOp

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import GradScaler, autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield

    GradScaler = None

try:
    import fairscale
except ImportError:
    fairscale = None


@dataclasses.dataclass
class GANTrainerOptions(TrainerOptions):
    """Trainer option dataclass for GANTrainer."""

    generator_first: bool


class GANTrainer(Trainer):
    """Trainer for GAN-based training.

    If you'd like to use this trainer, the model must inherit
    espnet.train.abs_gan_espnet_model.AbsGANESPnetModel.

    """

    @classmethod
    def build_options(cls, args: argparse.Namespace) -> TrainerOptions:
        """Build options consumed by train(), eval(), and plot_attention()."""
        assert check_argument_types()
        return build_dataclass(GANTrainerOptions, args)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add additional arguments for GAN-trainer."""
        parser.add_argument(
            "--generator_first",
            type=str2bool,
            default=False,
            help="Whether to update generator first.",
        )

    @staticmethod
    def _check_unsupported_options(options: GANTrainerOptions) -> None:
        """Reject option combinations not implemented for GAN training."""
        # TODO(kan-bayashi): Support the use of these options
        if options.accum_grad > 1:
            raise NotImplementedError(
                "accum_grad > 1 is not supported in GAN-based training."
            )
        if options.grad_noise:
            raise NotImplementedError(
                "grad_noise is not supported in GAN-based training."
            )

    @staticmethod
    def _turn_order(generator_first: bool) -> List[str]:
        """Return the ["generator", "discriminator"] update order for a batch."""
        if generator_first:
            return ["generator", "discriminator"]
        return ["discriminator", "generator"]

    @classmethod
    def train_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler],
        reporter: SubReporter,
        summary_writer,
        options: GANTrainerOptions,
        distributed_option: DistributedOption,
    ) -> bool:
        """Train one epoch, alternating generator/discriminator updates.

        For each minibatch, runs both a generator turn and a discriminator
        turn (see :meth:`_turn_order`), each with its own forward pass,
        backward pass, gradient clipping, and optimizer step.

        Returns:
            True if every turn in this epoch produced a non-finite gradient
            norm, i.e. no optimizer step ever happened.
        """
        assert check_argument_types()

        grad_clip = options.grad_clip
        grad_clip_type = options.grad_clip_type
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        use_wandb = options.use_wandb
        generator_first = options.generator_first
        distributed = distributed_option.distributed

        cls._check_unsupported_options(options)
        log_interval = cls._resolve_log_interval(iterator, options.log_interval)

        model.train()
        all_steps_are_invalid = True
        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        start_time = time.perf_counter()
        for iiter, (_, batch) in enumerate(
            reporter.measure_iter_time(iterator, "iter_time"), 1
        ):
            assert isinstance(batch, dict), type(batch)

            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                all_steps_are_invalid = False
                continue

            turn_start_time = time.perf_counter()
            for turn in cls._turn_order(generator_first):
                with autocast(scaler is not None):
                    with reporter.measure_time(f"{turn}_forward_time"):
                        loss, stats, weight, optim_idx = cls._forward_turn_loss(
                            model, batch, turn
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
                        # NOTE(kamo): Multiply world_size since DistributedDataParallel
                        # automatically normalizes the gradient by world_size.
                        loss *= torch.distributed.get_world_size()

                reporter.register(stats, weight)

                grad_was_valid = cls._backward_and_step_turn(
                    model=model,
                    loss=loss,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    scaler=scaler,
                    optim_idx=optim_idx,
                    grad_clip=grad_clip,
                    grad_clip_type=grad_clip_type,
                    turn=turn,
                    reporter=reporter,
                )
                if grad_was_valid:
                    all_steps_are_invalid = False

                reporter.register(
                    {f"{turn}_train_time": time.perf_counter() - turn_start_time}
                )
                turn_start_time = time.perf_counter()

            reporter.register({"train_time": time.perf_counter() - start_time})
            start_time = time.perf_counter()

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

    @classmethod
    def _forward_turn_loss(
        cls, model: torch.nn.Module, batch: Dict[str, torch.Tensor], turn: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Optional[int]]:
        """Run the forward pass for one generator/discriminator turn.

        The GAN model must return a dict with "loss"/"stats"/"weight"/
        "optim_idx" (0 for generator, 1 for discriminator).
        """
        retval = model(forward_generator=turn == "generator", **batch)

        # Note(kamo):
        # Supporting two patterns for the returned value from the model
        #   a. dict type
        if isinstance(retval, dict):
            loss = retval["loss"]
            stats = retval["stats"]
            weight = retval["weight"]
            optim_idx = cls._normalize_optim_idx(retval.get("optim_idx"))

        # b. tuple or list type
        else:
            raise RuntimeError("model output must be dict.")
        return loss, stats, weight, optim_idx

    @staticmethod
    def _backward_and_step_turn(
        model: torch.nn.Module,
        loss: torch.Tensor,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler],
        optim_idx: Optional[int],
        grad_clip: float,
        grad_clip_type: float,
        turn: str,
        reporter: SubReporter,
    ) -> bool:
        """Backward, clip, and step the optimizer owning this turn's loss.

        Unlike the base :class:`Trainer`, GAN training has no gradient
        accumulation: every turn backpropagates and steps immediately, and
        both optimizers' gradients are cleared after every turn.

        Returns:
            Whether the gradient norm was finite (an optimizer step
            actually happened). When ``grad_clip <= 0``, clipping (and
            thus the finiteness check) is skipped and this is always True.
        """
        with reporter.measure_time(f"{turn}_backward_time"):
            if scaler is not None:
                # Scales loss.  Calls backward() on scaled loss
                # to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose
                # for corresponding forward ops.
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler is not None:
            # Unscales the gradients of optimizer's assigned params in-place
            for iopt, optimizer in enumerate(optimizers):
                if optim_idx is not None and iopt != optim_idx:
                    continue
                scaler.unscale_(optimizer)

        # TODO(kan-bayashi): Compute grad norm without clipping
        grad_norm = None
        if grad_clip > 0.0:
            # compute the gradient norm to check if it is normal or not
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
                norm_type=grad_clip_type,
            )
            # PyTorch<=1.4, clip_grad_norm_ returns float value
            if not isinstance(grad_norm, torch.Tensor):
                grad_norm = torch.tensor(grad_norm)

        grad_was_valid = grad_norm is None or bool(torch.isfinite(grad_norm))
        if grad_was_valid:
            with reporter.measure_time(f"{turn}_optim_step_time"):
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
        else:
            logging.warning(f"The grad norm is {grad_norm}. Skipping updating the model.")
            # Must invoke scaler.update() if unscale_() is used in the
            # iteration to avoid the following error:
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

        for iopt, optimizer in enumerate(optimizers):
            # NOTE(kan-bayashi): In the case of GAN, we need to clear
            #   the gradient of both optimizers after every update.
            optimizer.zero_grad()

        # Register lr and train/load time[sec/step],
        # where step refers to accum_grad * mini-batch
        reporter.register(
            {
                f"optim{optim_idx}_lr{i}": pg["lr"]
                for i, pg in enumerate(optimizers[optim_idx].param_groups)
                if "lr" in pg
            },
        )
        return grad_was_valid

    @classmethod
    @torch.no_grad()
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: GANTrainerOptions,
        distributed_option: DistributedOption,
    ) -> None:
        """Validate one epoch: forward-only generator and discriminator turns."""
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = distributed_option.distributed
        generator_first = options.generator_first

        model.eval()

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        for _, batch in iterator:
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            for turn in cls._turn_order(generator_first):
                retval = model(forward_generator=turn == "generator", **batch)
                if isinstance(retval, dict):
                    stats = retval["stats"]
                    weight = retval["weight"]
                else:
                    _, stats, weight = retval
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
