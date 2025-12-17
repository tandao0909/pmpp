import torch
from torch import Tensor

__all__ = ["mymuladd", "myadd_out"]


def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    return torch.ops.extension_cpp.mymuladd.default(a, b, c)
