"""Gradient noise injection utility, applied to model gradients during training.

Adding decaying Gaussian noise to gradients is a regularization technique
(Neelakantan et al., 2015, "Adding Gradient Noise Improves Learning for Very
Deep Networks") that can help optimization escape sharp minima early in
training while annealing to zero noise as training progresses.
"""
import torch


def add_gradient_noise(
    model: torch.nn.Module,
    iteration: int,
    duration: float = 100,
    eta: float = 1.0,
    scale_factor: float = 0.55,
) -> None:
    """Add noise from a standard normal distribution to the gradients.

    The standard deviation (`sigma`) is controlled
    by the three hyper-parameters below.
    `sigma` goes to zero (no noise) with more iterations.

    Args:
        model: Model whose parameter gradients will be perturbed in place.
        iteration: Number of iterations (i.e. current training step).
        duration: {100, 1000}: Number of durations to control
            the interval of the `sigma` change.
        eta: {0.01, 0.3, 1.0}: The magnitude of `sigma`.
        scale_factor: {0.55}: The scale of `sigma`.
    """
    interval = (iteration // duration) + 1
    sigma = eta / interval**scale_factor
    for param in model.parameters():
        if param.grad is not None:
            grad_shape = param.grad.size()
            noise = sigma * torch.randn(grad_shape).to(param.device)
            param.grad += noise
