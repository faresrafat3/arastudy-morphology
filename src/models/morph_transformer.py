"""AraLingua Transformer with MorphologicalEmbedding."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.morphological_embedding import MorphEmbeddingConfig, MorphologicalEmbedding
from src.models.transformer import (  # type: ignore[import-untyped]
    ModelArgs,
    RMSNorm,
    TransformerBlock,
)


class AraLinguaTransformer(nn.Module):
    """AraStudy-like Transformer that supports compositional Arabic morphology."""

    def __init__(self, args: ModelArgs, morph_config: MorphEmbeddingConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size

        self.embedding = MorphologicalEmbedding(morph_config)

        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        root_ids: torch.Tensor | None = None,
        pattern_ids: torch.Tensor | None = None,
        affix_ids: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if root_ids is not None and pattern_ids is not None and affix_ids is not None:
            hidden_states = self.embedding(token_ids, root_ids, pattern_ids, affix_ids)
        else:
            hidden_states = self.embedding.bpe_fallback(token_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        loss: torch.Tensor | None = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss

    def morphological_regularization_loss(self) -> torch.Tensor:
        num_roots = self.embedding.config.num_roots
        num_samples = min(64, num_roots)
        device = self.embedding.root_embeddings.weight.device

        root_indices = torch.randint(0, num_roots, (num_samples,), device=device)
        root_vecs = self.embedding.root_embeddings(root_indices)

        t_shared = self.embedding.transform_shared_U @ self.embedding.transform_shared_V

        pattern_1 = torch.randint(
            0,
            self.embedding.config.num_patterns,
            (num_samples,),
            device=device,
        )
        pattern_2 = torch.randint(
            0,
            self.embedding.config.num_patterns,
            (num_samples,),
            device=device,
        )

        u1 = self.embedding.transform_U[pattern_1]
        v1 = self.embedding.transform_V[pattern_1]
        u2 = self.embedding.transform_U[pattern_2]
        v2 = self.embedding.transform_V[pattern_2]

        shared = root_vecs @ t_shared.T
        delta_1 = torch.einsum("nrd,nr->nd", u1, torch.einsum("nrd,nd->nr", v1, root_vecs))
        delta_2 = torch.einsum("nrd,nr->nd", u2, torch.einsum("nrd,nd->nr", v2, root_vecs))

        transformed_1 = shared + delta_1
        transformed_2 = shared + delta_2

        same_root_dist = torch.mean((transformed_1 - transformed_2) ** 2)

        shuffled = transformed_2[torch.randperm(num_samples, device=device)]
        diff_root_dist = torch.mean((transformed_1 - shuffled) ** 2)

        return torch.relu(same_root_dist - diff_root_dist + 0.5)
