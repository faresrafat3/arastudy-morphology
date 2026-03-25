"""AraStudy decoder-only transformer (RMSNorm + RoPE + SwiGLU)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


@dataclass
class ModelArgs:
	dim: int = 512
	n_layers: int = 6
	n_heads: int = 8
	vocab_size: int = 16000
	max_seq_len: int = 512
	dropout: float = 0.1
	norm_eps: float = 1e-5


def _round_to_multiple(value: int, multiple: int) -> int:
	return ((value + multiple - 1) // multiple) * multiple


class RMSNorm(nn.Module):
	def __init__(self, dim: int, eps: float = 1e-5):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(dim))
		self.eps = eps

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		variance = x.pow(2).mean(dim=-1, keepdim=True)
		x = x * torch.rsqrt(variance + self.eps)
		return self.weight * x


class RotaryEmbedding(nn.Module):
	def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
		super().__init__()
		self.head_dim = head_dim
		freqs_cis = self.precompute_freqs_cis(head_dim, max_seq_len, base)
		self.register_buffer("freqs_cis", freqs_cis, persistent=False)

	@staticmethod
	def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
		if dim % 2 != 0:
			raise ValueError(f"RoPE dim must be even, got {dim}")
		freq_seq = torch.arange(0, dim, 2, dtype=torch.float32)
		inv_freq = 1.0 / (base ** (freq_seq / dim))
		t = torch.arange(max_seq_len, dtype=torch.float32)
		freqs = torch.outer(t, inv_freq)
		return torch.polar(torch.ones_like(freqs), freqs)

	@staticmethod
	def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
		return freqs_cis.unsqueeze(0).unsqueeze(2).to(x.device)

	@staticmethod
	def apply_rotary_emb(
		xq: torch.Tensor,
		xk: torch.Tensor,
		freqs_cis: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor]:
		xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
		xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
		freqs = RotaryEmbedding._reshape_for_broadcast(freqs_cis, xq_)
		xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
		xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)
		return xq_out.type_as(xq), xk_out.type_as(xk)

	def forward(self, xq: torch.Tensor, xk: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
		freqs_cis = self.freqs_cis[:seq_len]
		return self.apply_rotary_emb(xq, xk, freqs_cis)


class Attention(nn.Module):
	def __init__(self, args: ModelArgs):
		super().__init__()
		if args.dim % args.n_heads != 0:
			raise ValueError("dim must be divisible by n_heads")
		self.n_heads = args.n_heads
		self.head_dim = args.dim // args.n_heads
		self.dropout = args.dropout
		self.wq = nn.Linear(args.dim, args.dim, bias=False)
		self.wk = nn.Linear(args.dim, args.dim, bias=False)
		self.wv = nn.Linear(args.dim, args.dim, bias=False)
		self.wo = nn.Linear(args.dim, args.dim, bias=False)
		self.rope = RotaryEmbedding(self.head_dim, args.max_seq_len)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_size, seq_len, _ = x.shape
		q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
		k = self.wk(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
		v = self.wv(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

		q, k = self.rope(q, k, seq_len)

		q = q.transpose(1, 2)
		k = k.transpose(1, 2)
		v = v.transpose(1, 2)

		attn = F.scaled_dot_product_attention(
			q,
			k,
			v,
			attn_mask=None,
			dropout_p=self.dropout if self.training else 0.0,
			is_causal=True,
		)
		attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
		return self.wo(attn)


class SwiGLUFFN(nn.Module):
	def __init__(self, dim: int, dropout: float):
		super().__init__()
		hidden_dim = _round_to_multiple(int((2 * 4 * dim) / 3), 256)
		self.w1 = nn.Linear(dim, hidden_dim, bias=False)
		self.w2 = nn.Linear(dim, hidden_dim, bias=False)
		self.w3 = nn.Linear(hidden_dim, dim, bias=False)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		gate = self.w1(x)
		up = self.w2(x)
		x = F.silu(gate) * up
		x = self.w3(x)
		return self.dropout(x)


class TransformerBlock(nn.Module):
	def __init__(self, args: ModelArgs):
		super().__init__()
		self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
		self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
		self.attention = Attention(args)
		self.ffn = SwiGLUFFN(args.dim, args.dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.attention(self.attention_norm(x))
		x = x + self.ffn(self.ffn_norm(x))
		return x


class AraStudyTransformer(nn.Module):
	def __init__(self, args: ModelArgs):
		super().__init__()
		self.args = args
		self.max_seq_len = args.max_seq_len
		self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
		self.dropout = nn.Dropout(args.dropout)
		self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
		self.norm = RMSNorm(args.dim, eps=args.norm_eps)
		self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

		self.apply(self._init_weights)
		self.output.weight = self.tok_embeddings.weight

	def _init_weights(self, module: nn.Module) -> None:
		if isinstance(module, (nn.Linear, nn.Embedding)):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if isinstance(module, nn.Linear) and module.bias is not None:
				nn.init.zeros_(module.bias)

	def forward(
		self,
		token_ids: torch.Tensor,
		targets: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor | None]:
		if token_ids.size(1) > self.max_seq_len:
			raise ValueError(
				f"Sequence length {token_ids.size(1)} exceeds max_seq_len {self.max_seq_len}"
			)

		x = self.tok_embeddings(token_ids)
		x = self.dropout(x)

		for layer in self.layers:
			x = layer(x)

		x = self.norm(x)
		logits = self.output(x)

		loss = None
		if targets is not None:
			loss = F.cross_entropy(
				logits.view(-1, logits.size(-1)),
				targets.reshape(-1),
			)
		return logits, loss

	def forward_hidden(self, token_ids: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
		x = self.tok_embeddings(token_ids)
		x = self.dropout(x)
		hidden_states: list[torch.Tensor] = []
		for layer in self.layers:
			x = layer(x)
			hidden_states.append(x)
		x = self.norm(x)
		return hidden_states, x

	@classmethod
	def from_config(cls, config_path: str, vocab_size: int) -> "AraStudyTransformer":
		return from_config(config_path, vocab_size)


def from_config(config_path: str, vocab_size: int) -> AraStudyTransformer:
	with open(config_path, encoding="utf-8") as handle:
		cfg = yaml.safe_load(handle)
	model_cfg: dict[str, Any] = cfg.get("model", cfg)
	args = ModelArgs(
		dim=int(model_cfg.get("dim", 512)),
		n_layers=int(model_cfg.get("n_layers", 6)),
		n_heads=int(model_cfg.get("n_heads", 8)),
		vocab_size=vocab_size,
		max_seq_len=int(model_cfg.get("max_seq_len", 512)),
		dropout=float(model_cfg.get("dropout", 0.1)),
		norm_eps=float(model_cfg.get("norm_eps", 1e-5)),
	)
	return AraStudyTransformer(args)


def count_parameters(model: nn.Module) -> dict[str, int]:
	embedding = 0
	attention = 0
	ffn = 0
	norm = 0
	total = 0

	for name, parameter in model.named_parameters():
		count = parameter.numel()
		total += count
		lower = name.lower()
		if "tok_embeddings" in lower or lower.startswith("output"):
			embedding += count
		elif any(key in lower for key in ["attention", ".wq", ".wk", ".wv", ".wo"]):
			attention += count
		elif any(key in lower for key in ["ffn", ".w1", ".w2", ".w3"]):
			ffn += count
		elif "norm" in lower:
			norm += count

	return {
		"embedding": embedding,
		"attention": attention,
		"ffn": ffn,
		"norm": norm,
		"total": total,
	}
