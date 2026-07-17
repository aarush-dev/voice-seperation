"""Adaptor that exposes an arbitrary module method as `forward()`.

Used to make non-`forward` methods compatible with `torch.nn.DataParallel`,
which only parallelizes calls to a module's `forward()`.
"""
from typing import Any

import torch
from typeguard import check_argument_types


class ForwardAdaptor(torch.nn.Module):
    """Wrap a module so that a chosen method is invoked as `forward()`.

    `torch.nn.DataParallel` parallelizes only "forward()"
    and, maybe, the method having the other name can't be applied
    except for wrapping the module just like this class.

    Examples:
        >>> class A(torch.nn.Module):
        ...     def foo(self, x):
        ...         ...
        >>> model = A()
        >>> model = ForwardAdaptor(model, "foo")
        >>> model = torch.nn.DataParallel(model, device_ids=[0, 1])
        >>> x = torch.randn(2, 10)
        >>> model(x)
    """

    def __init__(self, module: torch.nn.Module, name: str):
        """Store `module` and remember which of its methods to forward to.

        Args:
            module: The wrapped module.
            name: Name of the attribute/method on `module` to call from
                `forward()`.

        Raises:
            ValueError: If `module` has no attribute named `name`.
        """
        assert check_argument_types()
        super().__init__()
        self.module = module
        self.name = name
        if not hasattr(module, name):
            raise ValueError(f"{module} doesn't have {name}")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Call `self.module.<name>(*args, **kwargs)`."""
        func = getattr(self.module, self.name)
        return func(*args, **kwargs)
