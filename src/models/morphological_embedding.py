"""Linear Morphological Transform Embedding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MorphEmbeddingConfig:
    vocab_size: int = 16384
    dim: int = 512
    num_roots: int = 3000
    num_patterns: int = 21
    num_affixes: int = 100
    root_dim: int = 128
    transform_rank: int = 16
    num_function_words: int = 200
    morph_coverage: float = 0.0


class MorphologicalEmbedding(nn.Module):
    """Compositional embedding with root + pattern transforms and BPE fallback."""

    def __init__(self, config: MorphEmbeddingConfig):
        super().__init__()
        self.config = config

        self.root_embeddings = nn.Embedding(config.num_roots + 1, config.root_dim)

        self.transform_U = nn.Parameter(
            torch.randn(config.num_patterns, config.root_dim, config.transform_rank)
            * 0.02
        )
        self.transform_V = nn.Parameter(
            torch.randn(config.num_patterns, config.transform_rank, config.root_dim)
            * 0.02
        )
        self.transform_shared_U = nn.Parameter(
            torch.randn(config.root_dim, config.transform_rank) * 0.02
        )
        self.transform_shared_V = nn.Parameter(
            torch.randn(config.transform_rank, config.root_dim) * 0.02
        )

        self.affix_embeddings = nn.Embedding(config.num_affixes + 1, config.root_dim // 2)

        self.morph_projection = nn.Sequential(
            nn.Linear(config.root_dim + config.root_dim // 2, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.dim),
        )

        self.bpe_fallback = nn.Embedding(config.vocab_size, config.dim)
        self.gate_bias = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.root_embeddings.weight, std=0.02)
        nn.init.normal_(self.bpe_fallback.weight, std=0.02)
        nn.init.normal_(self.affix_embeddings.weight, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        root_ids: torch.Tensor,
        pattern_ids: torch.Tensor,
        affix_ids: torch.Tensor,
    ) -> torch.Tensor:
        bpe_emb = self.bpe_fallback(token_ids)

        has_morph = (root_ids >= 0) & (pattern_ids >= 0)
        if not has_morph.any():
            return bpe_emb

        safe_root_ids = root_ids.clamp(min=0, max=self.config.num_roots)
        safe_pattern_ids = pattern_ids.clamp(min=0, max=self.config.num_patterns - 1)
        safe_affix_ids = affix_ids.clamp(min=0, max=self.config.num_affixes)

        root_vecs = self.root_embeddings(safe_root_ids)

        t_shared = self.transform_shared_U @ self.transform_shared_V
        shared_transformed = F.linear(root_vecs, t_shared)

        selected_u = self.transform_U[safe_pattern_ids]
        selected_v = self.transform_V[safe_pattern_ids]
        intermediate = torch.einsum("bsrd,bsd->bsr", selected_v, root_vecs)
        delta_transformed = torch.einsum("bsdr,bsr->bsd", selected_u, intermediate)

        morph_vec = shared_transformed + delta_transformed

        affix_vec = self.affix_embeddings(safe_affix_ids)
        morph_full = torch.cat([morph_vec, affix_vec], dim=-1)
        morph_emb = self.morph_projection(morph_full)

        gate = torch.sigmoid(self.gate_bias)
        return torch.where(
            has_morph.unsqueeze(-1),
            gate * morph_emb + (1.0 - gate) * bpe_emb,
            bpe_emb,
        )

    def count_parameters(self) -> dict[str, int]:
        return {
            "root_embeddings": self.root_embeddings.weight.numel(),
            "transforms": (
                self.transform_U.numel()
                + self.transform_V.numel()
                + self.transform_shared_U.numel()
                + self.transform_shared_V.numel()
            ),
            "affix_embeddings": self.affix_embeddings.weight.numel(),
            "morph_projection": sum(
                parameter.numel() for parameter in self.morph_projection.parameters()
            ),
            "bpe_fallback": self.bpe_fallback.weight.numel(),
            "gate": 1,
            "total": sum(parameter.numel() for parameter in self.parameters()),
        }
