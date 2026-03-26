#!/usr/bin/env python3
'''
AraStudy Transformer + Root Embedding.
Phase 2: emb = tok_emb + root_emb
'''

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.transformer import AraStudyTransformer, ModelArgs


class RootEmbeddingTransformer(AraStudyTransformer):
    '''Transformer with additional root embedding.

    emb(token) = tok_embedding(token_id) + root_embedding(root_id)

    root_id comes from pre-computed morphological analysis.
    '''

    def __init__(self, args: ModelArgs, num_roots: int = 5000):
        super().__init__(args)
        self.num_roots = num_roots
        self.root_embeddings = nn.Embedding(num_roots + 1, args.dim)
        nn.init.normal_(self.root_embeddings.weight, std=0.01)
        self.root_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        root_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.tok_embeddings(token_ids)

        if root_ids is not None:
            root_emb = self.root_embeddings(root_ids)
            x = x + self.root_scale * root_emb

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.output(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss
