"""Baseline transformer placeholder.

Model implementation intentionally deferred.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelArgs:
	dim: int = 512
	n_layers: int = 6
	n_heads: int = 8
	vocab_size: int = 16000
	max_seq_len: int = 512
	dropout: float = 0.1
	norm_eps: float = 1e-5


class RMSNorm(nn.Module):
	def __init__(self, dim: int, eps: float = 1e-5):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(dim))
		self.eps = eps

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		variance = x.pow(2).mean(dim=-1, keepdim=True)
		x = x * torch.rsqrt(variance + self.eps)
		return self.weight * x


class TransformerBlock(nn.Module):
	def __init__(self, _args: ModelArgs):
		super().__init__()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x
