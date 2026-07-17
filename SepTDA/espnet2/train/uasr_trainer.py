# Copyright 2022 Tomoki Hayashi
#           2022 Dongji Gao
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Trainer module for GAN-based UASR training.

:class:`UASRTrainer` is the UASR (Unsupervised ASR) counterpart of
:class:`espnet2.train.gan_trainer.GANTrainer`: it reuses
:class:`espnet2.train.trainer.Trainer`'s ``run()`` loop but each minibatch
runs a single generator-or-discriminator turn (chosen by the model itself
via ``is_discriminative_step()``) rather than always alternating both.
Validation additionally computes a phone error rate and, if the model has a
KenLM language model attached, a vocabulary-weighted LM perplexity.
"""

import argparse
import dataclasses
import logging
import math
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
class UASRTrainerOptions(TrainerOptions):
    """Trainer option dataclass for UASRTrainer."""

    generator_first: bool
    max_num_warning: int


class UASRTrainer(Trainer):
    """Trainer for GAN-based UASR training.

    If you'd like to use this trainer, the model must inherit
    espnet.train.abs_gan_espnet_model.AbsGANESPnetModel.

    """

    @classmethod
    def build_options(cls, args: argparse.Namespace) -> TrainerOptions:
        """Build options consumed by train(), eval(), and plot_attention()."""
        assert check_argument_types()
        return build_dataclass(UASRTrainerOptions, args)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add additional arguments for GAN-trainer."""
        parser.add_argument(
            "--generator_first",
            type=str2bool,
            default=False,
            help="Whether to update generator first.",
        )
        parser.add_argument(
            "--max_num_warning",
            type=int,
            default=10,
            help="Maximum number of warning shown",
        )

    @staticmethod
    def _check_unsupported_options(options: UASRTrainerOptions) -> None:
        """Reject option combinations not implemented for GAN-UASR training."""
        # TODO(kan-bayashi): Support the use of these options
        if options.accum_grad > 1:
            raise NotImplementedError(
                "accum_grad > 1 is not supported in GAN-based training."
            )
        if options.grad_noise:
            raise NotImplementedError(
                "grad_noise is not supported in GAN-based training."
            )

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
        options: UASRTrainerOptions,
        distributed_option: DistributedOption,
    ) -> bool:
        """Train one epoch, letting the model choose generator vs. discriminator.

        Unlike :class:`GANTrainer`, each minibatch runs a single turn:
        ``model.is_discriminative_step()`` decides whether this step
        updates the discriminator or the generator, and
        ``model.get_optim_index()`` selects the matching optimizer.
        Repeated "invalid gradient" warnings are folded after
        ``options.max_num_warning`` occurrences to avoid flooding the log.

        Returns:
            True if every step in this epoch produced a non-finite
            gradient norm, i.e. no optimizer step ever happened.
        """
        assert check_argument_types()

        grad_clip = options.grad_clip
        grad_clip_type = options.grad_clip_type
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        use_wandb = options.use_wandb
        distributed = distributed_option.distributed
        max_num_warning = options.max_num_warning
        cur_num_warning = 0
        hide_warning = False

        cls._check_unsupported_options(options)
        log_interval = cls._resolve_log_interval(iterator, options.log_interval)

        model.train()
        all_steps_are_invalid = True
        model.number_epochs = reporter.epoch
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
            model.number_updates = iiter - 1
            is_discriminative_step = model.is_discriminative_step()
            optim_idx = model.get_optim_index()

            turn = "discriminator" if is_discriminative_step else "generator"
            with autocast(scaler is not None):
                with reporter.measure_time(f"{turn}_forward_time"):
                    loss, stats, weight = cls._forward_turn_loss(model, batch)

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

            grad_was_valid, grad_norm = cls._apply_turn_gradient_step(
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
            else:
                cur_num_warning += 1
                if cur_num_warning >= max_num_warning:
                    if not hide_warning:
                        logging.info("Warning info folded...")
                    hide_warning = True

                if not hide_warning:
                    logging.warning(
                        f"The grad norm is {grad_norm}. "
                        "Skipping updating the model."
                    )

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

        if hide_warning:
            logging.warning(
                f"{cur_num_warning}/{iiter} iterations skipped due to inf/nan grad norm"
            )

        return all_steps_are_invalid

    @staticmethod
    def _forward_turn_loss(
        model: torch.nn.Module, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Run the forward pass; the UASR model must return a tuple/list."""
        retval = model(**batch)

        # Note(jiatong):
        # Supporting only one patterns for the returned value from the model
        # must be tuple or list type
        if not (isinstance(retval, list) or isinstance(retval, tuple)):
            raise RuntimeError("model output must be tuple or list.")
        loss, stats, weight, _ = retval
        return loss, stats, weight

    @staticmethod
    def _apply_turn_gradient_step(
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
    ) -> Tuple[bool, Optional[torch.Tensor]]:
        """Backward, clip, and step the optimizer owning this turn's loss.

        No gradient accumulation: backpropagates and steps immediately,
        clearing every optimizer's gradients afterward. The caller (not
        this helper) decides whether to log an invalid-gradient warning,
        since :meth:`UASRTrainer.train_one_epoch` folds repeated warnings.

        Returns:
            (grad_was_valid, grad_norm)
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
        return grad_was_valid, grad_norm

    @classmethod
    @torch.no_grad()
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: UASRTrainerOptions,
        distributed_option: DistributedOption,
    ) -> None:
        """Validate one epoch, then compute phone error rate and (if a KenLM
        language model is attached) a vocabulary-weighted LM perplexity."""
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = distributed_option.distributed

        vocab_seen_list = []

        model.eval()

        logging.info("Doing validation")
        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        print_hyp = True
        for _, batch in iterator:
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            retval = model(**batch, do_validation=True, print_hyp=print_hyp)
            print_hyp = False
            if not (isinstance(retval, list) or isinstance(retval, tuple)):
                raise RuntimeError("model output must be tuple or list.")
            else:
                loss, stats, weight, vocab_seen = retval
                vocab_seen_list.append(vocab_seen)

            stats = {k: v for k, v in stats.items() if v is not None}

            if ngpu > 1 or distributed:
                # Apply weighted averaging for stats.
                # if distributed, this method can also apply all_reduce()
                stats, weight = recursive_average(stats, weight, distributed)

            reporter.register(stats)
            reporter.next()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

        reporter.register({"phone_error_rate": cls._compute_phone_error_rate(reporter)})

        if model.kenlm:
            weighted_lm_ppl, lm_ppl = cls._compute_weighted_lm_ppl(
                reporter, model, vocab_seen_list
            )
            reporter.register({"lm_ppl": lm_ppl})
            reporter.register({"weighted_lm_ppl": weighted_lm_ppl})

        reporter.next()

    @staticmethod
    def _compute_phone_error_rate(reporter: SubReporter) -> float:
        """Aggregate per-batch error/ref-token counts into a phone error rate."""
        total_num_errors = 0
        total_num_ref_tokens = 0
        assert (
            "batch_num_errors" in reporter.stats
            and "batch_num_ref_tokens" in reporter.stats
        )
        for batch_num_errors, batch_num_ref_tokens in zip(
            reporter.stats["batch_num_errors"], reporter.stats["batch_num_ref_tokens"]
        ):
            total_num_errors += batch_num_errors.value
            total_num_ref_tokens += batch_num_ref_tokens.value
        return total_num_errors / total_num_ref_tokens

    @staticmethod
    def _compute_weighted_lm_ppl(
        reporter: SubReporter, model: torch.nn.Module, vocab_seen_list: List[torch.Tensor]
    ) -> Tuple[float, float]:
        """Compute the KenLM perplexity and its vocabulary-coverage-weighted variant.

        Returns:
            (weighted_lm_ppl, lm_ppl)
        """
        assert (
            "batch_lm_log_prob" in reporter.stats
            and "batch_num_hyp_tokens" in reporter.stats
            and "batch_size" in reporter.stats
        )
        assert (
            len(reporter.stats["batch_lm_log_prob"])
            == len(reporter.stats["batch_num_hyp_tokens"])
            == len(reporter.stats["batch_size"])
        )

        total_lm_log_prob = 0
        total_num_tokens = 0
        total_num_sentences = 0
        for log_prob, num_tokens, batch_size in zip(
            reporter.stats["batch_lm_log_prob"],
            reporter.stats["batch_num_hyp_tokens"],
            reporter.stats["batch_size"],
        ):
            total_lm_log_prob += log_prob.value
            total_num_tokens += num_tokens.value
            total_num_sentences += batch_size.value
        lm_ppl = math.pow(
            10, -total_lm_log_prob / (total_num_tokens + total_num_sentences)
        )

        vocab_seen = torch.stack(vocab_seen_list).sum(dim=0).bool().sum()
        vocab_seen_rate = vocab_seen / model.vocab_size
        assert vocab_seen_rate <= 1.0
        weighted_lm_ppl = lm_ppl / vocab_seen_rate**2

        return weighted_lm_ppl, lm_ppl
