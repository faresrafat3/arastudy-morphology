# Morphology Roadmap (Revised after critical review)

## Goal
Improve Arabic morphology awareness with progressive, test-first steps.

## Phase 0 — Problem Confirmation (1 day)
1. Train baseline model as-is.
2. Extract token embedding matrix from checkpoint.
3. Probe whether same-root words are closer than random pairs.
4. Decision gate:
   - If same-root proximity is already strong, architecture changes are lower priority.
   - If weak, continue to Phase 1.

### Scripts
- `scripts/extract_baseline_embeddings.py`
- `scripts/phase0_probe_root_similarity.py`

### Example
```bash
python scripts/extract_baseline_embeddings.py \
  --checkpoint results/exp01/run_x/best.pt \
  --sp-model results/tokenizers/phase2b/bpe_16k.model \
  --output-dir results/analysis/phase0

python scripts/phase0_probe_root_similarity.py \
  --embeddings results/analysis/phase0/token_embeddings.npy \
  --tokens results/analysis/phase0/tokens.txt \
  --output results/analysis/phase0/root_probe.json
```

## Phase 1 — Data Only (1 week)
No architecture change.

1. Generate morphology root-list lines (e.g., `جذر كتب: كتب كاتب مكتوب ...`).
2. Mix these lines into training corpus.
3. Retrain baseline and compare with baseline-only model.

### Script
- `scripts/build_morph_data_only_corpus.py`

## Phase 2 — Lightweight Architecture (2 weeks)
Use additive root feature only:

`embedding = token_embedding + root_embedding`

- Keep standard transformer design and weight tying where possible.
- Avoid heavy compositional transforms at this stage.

## Phase 3 — Advanced Compositional Morphology (optional)
Only if Phase 2 improvements are insufficient.

- Consider transform-based/group-theory-style embeddings.
- Add complexity only with evidence from previous phases.

## De-scoped for now
- Hierarchical decoding
- Matryoshka embeddings
- Complex multi-module pipeline
- Full transform-heavy morphology path in early phase
