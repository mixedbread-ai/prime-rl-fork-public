import torch
from torch import nn


class Qwen4ExpRMSNorm(nn.Module):
    def __init__(self, dim: int, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        if group_size is not None and dim % group_size != 0:
            raise ValueError(f"dim ({dim}) must be divisible by group_size ({group_size})")
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_size is None:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        grouped = x.unflatten(-1, (-1, self.group_size))
        return (grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + self.eps)).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()) * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, group_size={self.group_size}, eps={self.eps}"
