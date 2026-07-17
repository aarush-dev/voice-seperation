"""Reports the installed PyTorch/CUDA/cuDNN versions and settings, for logging."""
import torch


def pytorch_cudnn_version() -> str:
    """Build a one-line summary of the PyTorch/CUDA/cuDNN environment.

    Returns:
        A string with the PyTorch version, CUDA availability, and (when
        cuDNN is enabled) the cuDNN version plus its benchmark/deterministic
        flags.
    """
    message = (
        f"pytorch.version={torch.__version__}, "
        f"cuda.available={torch.cuda.is_available()}, "
    )

    if torch.backends.cudnn.enabled:
        message += (
            f"cudnn.version={torch.backends.cudnn.version()}, "
            f"cudnn.benchmark={torch.backends.cudnn.benchmark}, "
            f"cudnn.deterministic={torch.backends.cudnn.deterministic}"
        )
    return message
