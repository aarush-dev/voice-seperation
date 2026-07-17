"""Complex-tensor helpers that transparently support two complex backends.

ESPnet's enhancement code historically represented complex tensors with the
third-party ``torch_complex.ComplexTensor`` class (a pair of real tensors),
because older versions of PyTorch had no native complex dtype. Newer PyTorch
versions provide native complex tensors (``torch.is_complex(x) is True``).

Every function in this module accepts either representation and dispatches to
the matching implementation (``torch_complex.functional`` for
``ComplexTensor``, plain ``torch`` ops for native complex tensors), so callers
elsewhere in ``espnet2.enh`` do not need to know which backend is in use.

NOTE: Do not mix the two representations (a ``ComplexTensor`` and a native
complex ``torch.Tensor``) within a single call - several of the underlying ops
(``torch.einsum``, ``torch.matmul`` before PyTorch 1.9) do not support it.
"""

from typing import Sequence, Tuple, Union

import torch
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

EPS = torch.finfo(torch.double).eps


def new_complex_like(
    ref: Union[torch.Tensor, ComplexTensor],
    real_imag: Tuple[torch.Tensor, torch.Tensor],
) -> Union[torch.Tensor, ComplexTensor]:
    """Build a complex tensor from ``(real, imag)`` using the same backend as `ref`.

    Args:
        ref: A tensor whose type (``ComplexTensor`` vs. native complex)
            determines which backend the output uses.
        real_imag: ``(real, imag)`` pair of real-valued tensors of identical shape.

    Returns:
        A complex tensor of the same backend as ``ref``, shaped like the inputs.
    """
    if isinstance(ref, ComplexTensor):
        return ComplexTensor(*real_imag)
    elif is_torch_complex_tensor(ref):
        return torch.complex(*real_imag)
    else:
        raise ValueError(
            "Please update your PyTorch version to 1.9+ for complex support."
        )


def is_torch_complex_tensor(c) -> bool:
    """True if `c` is a native ``torch`` complex tensor (not ``ComplexTensor``)."""
    return not isinstance(c, ComplexTensor) and torch.is_complex(c)


def is_complex(c) -> bool:
    """True if `c` is complex-valued, in either supported backend."""
    return isinstance(c, ComplexTensor) or is_torch_complex_tensor(c)


def to_complex(c):
    """Convert `c` to a native ``torch`` complex tensor."""
    if isinstance(c, ComplexTensor):
        c = c.real + 1j * c.imag
        return c
    elif torch.is_complex(c):
        return c
    else:
        return torch.view_as_complex(c)


def to_double(c):
    """Cast a complex tensor of either backend to double precision."""
    if not isinstance(c, ComplexTensor) and torch.is_complex(c):
        return c.to(dtype=torch.complex128)
    else:
        return c.double()


def to_float(c):
    """Cast a complex tensor of either backend to single precision."""
    if not isinstance(c, ComplexTensor) and torch.is_complex(c):
        return c.to(dtype=torch.complex64)
    else:
        return c.float()


def cat(seq: Sequence[Union[ComplexTensor, torch.Tensor]], *args, **kwargs):
    """Backend-dispatching equivalent of ``torch.cat`` for a sequence of tensors."""
    if not isinstance(seq, (list, tuple)):
        raise TypeError(
            "cat(): argument 'tensors' (position 1) must be tuple of Tensors, "
            "not Tensor"
        )
    if isinstance(seq[0], ComplexTensor):
        return FC.cat(seq, *args, **kwargs)
    else:
        return torch.cat(seq, *args, **kwargs)


def complex_norm(
    c: Union[torch.Tensor, ComplexTensor], dim=-1, keepdim=False
) -> torch.Tensor:
    """Compute the L2 norm ``sqrt(sum(|c|^2))`` of a complex tensor along `dim`.

    For the ``ComplexTensor`` backend, ``EPS`` (double-precision machine
    epsilon) is added inside the square root to avoid a zero-gradient NaN at
    the origin; the native-complex path relies on ``torch.norm`` directly.
    """
    if not is_complex(c):
        raise TypeError("Input is not a complex tensor.")
    if is_torch_complex_tensor(c):
        return torch.norm(c, dim=dim, keepdim=keepdim)
    else:
        if dim is None:
            return torch.sqrt((c.real**2 + c.imag**2).sum() + EPS)
        else:
            return torch.sqrt(
                (c.real**2 + c.imag**2).sum(dim=dim, keepdim=keepdim) + EPS
            )


def einsum(equation: str, *operands):
    """Backend-dispatching equivalent of ``torch.einsum``.

    Accepts either a single sequence of operands or the operands passed
    directly (mirroring ``torch.einsum``'s calling conventions), and routes
    to ``torch_complex.functional.einsum`` or ``torch.einsum`` depending on
    which backend the operands use. When exactly one operand is real and the
    other is native complex, the einsum is applied separately to the real and
    imaginary parts (since ``torch.einsum`` does not support mixed
    real/complex operands on older PyTorch).
    """
    if len(operands) == 1:
        if isinstance(operands[0], (tuple, list)):
            operands = operands[0]
        complex_module = FC if isinstance(operands[0], ComplexTensor) else torch
        return complex_module.einsum(equation, *operands)
    elif len(operands) != 2:
        op0 = operands[0]
        same_type = all(op.dtype == op0.dtype for op in operands[1:])
        if same_type:
            _einsum = FC.einsum if isinstance(op0, ComplexTensor) else torch.einsum
            return _einsum(equation, *operands)
        else:
            raise ValueError("0 or More than 2 operands are not supported.")
    a, b = operands
    if isinstance(a, ComplexTensor) or isinstance(b, ComplexTensor):
        return FC.einsum(equation, a, b)
    elif torch.is_complex(a) or torch.is_complex(b):
        if not torch.is_complex(a):
            o_real = torch.einsum(equation, a, b.real)
            o_imag = torch.einsum(equation, a, b.imag)
            return torch.complex(o_real, o_imag)
        elif not torch.is_complex(b):
            o_real = torch.einsum(equation, a.real, b)
            o_imag = torch.einsum(equation, a.imag, b)
            return torch.complex(o_real, o_imag)
        else:
            return torch.einsum(equation, a, b)
    else:
        return torch.einsum(equation, a, b)


def inverse(
    c: Union[torch.Tensor, ComplexTensor],
) -> Union[torch.Tensor, ComplexTensor]:
    """Batched matrix inverse of a complex tensor, shape (..., N, N) -> (..., N, N)."""
    if isinstance(c, ComplexTensor):
        return c.inverse2()
    else:
        return c.inverse()


def matmul(
    a: Union[torch.Tensor, ComplexTensor], b: Union[torch.Tensor, ComplexTensor]
) -> Union[torch.Tensor, ComplexTensor]:
    """Backend-dispatching equivalent of ``torch.matmul`` for complex tensors.

    Supports mixing one real and one native-complex operand by applying the
    matmul to the real and imaginary parts separately (``torch.matmul`` does
    not support mixed real/complex operands on older PyTorch).
    """
    if isinstance(a, ComplexTensor) or isinstance(b, ComplexTensor):
        return FC.matmul(a, b)
    elif torch.is_complex(a) or torch.is_complex(b):
        if not torch.is_complex(a):
            o_real = torch.matmul(a, b.real)
            o_imag = torch.matmul(a, b.imag)
            return torch.complex(o_real, o_imag)
        elif not torch.is_complex(b):
            o_real = torch.matmul(a.real, b)
            o_imag = torch.matmul(a.imag, b)
            return torch.complex(o_real, o_imag)
        else:
            return torch.matmul(a, b)
    else:
        return torch.matmul(a, b)


def trace(a: Union[torch.Tensor, ComplexTensor]):
    """Batched matrix trace, shape (..., N, N) -> (...).

    Always routed through ``torch_complex.functional.trace`` because (as of
    PyTorch 1.9.0) ``torch.trace`` does not support batched inputs.
    """
    return FC.trace(a)


def reverse(a: Union[torch.Tensor, ComplexTensor], dim=0):
    """Reverse a complex tensor along `dim`."""
    if isinstance(a, ComplexTensor):
        return FC.reverse(a, dim=dim)
    else:
        return torch.flip(a, dims=(dim,))


def solve(b: Union[torch.Tensor, ComplexTensor], a: Union[torch.Tensor, ComplexTensor]):
    """Solve the linear equation ``a @ x = b`` for `x`.

    When both operands share the ``ComplexTensor`` backend, delegates to
    ``torch_complex.functional.solve``. When both are native complex tensors,
    uses ``torch.linalg.solve`` directly. Mixed real/complex operand pairs
    fall back to explicit ``inverse(a) @ b`` (matrix solve does not support
    mixed dtypes on older PyTorch).
    """
    # NOTE: Do not mix ComplexTensor and torch.complex in the input!
    if isinstance(a, ComplexTensor) or isinstance(b, ComplexTensor):
        if isinstance(a, ComplexTensor) and isinstance(b, ComplexTensor):
            return FC.solve(b, a, return_LU=False)
        else:
            return matmul(inverse(a), b)
    elif torch.is_complex(a) or torch.is_complex(b):
        if torch.is_complex(a) and torch.is_complex(b):
            return torch.linalg.solve(a, b)
        else:
            return matmul(inverse(a), b)
    else:
        return torch.linalg.solve(a, b)


def stack(seq: Sequence[Union[ComplexTensor, torch.Tensor]], *args, **kwargs):
    """Backend-dispatching equivalent of ``torch.stack`` for a sequence of tensors."""
    if not isinstance(seq, (list, tuple)):
        raise TypeError(
            "stack(): argument 'tensors' (position 1) must be tuple of Tensors, "
            "not Tensor"
        )
    if isinstance(seq[0], ComplexTensor):
        return FC.stack(seq, *args, **kwargs)
    else:
        return torch.stack(seq, *args, **kwargs)
