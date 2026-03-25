# AraStudy Literature Review (Top 10)

## 1) CAMeLBERT (Inoue et al., 2021)
- Arabic BERT variants and probing relevance.
- Gap for AraStudy: no explicit morphology-aware training objective for small decoder-only LMs.

## 2) Jais (Sengupta et al., 2023)
- Arabic-English bilingual LLM and Arabic tokenizer considerations.
- Gap: no explicit root-aware embedding mechanism.

## 3) TinyStories (Eldan & Li, 2023)
- Small LM performance from constrained but curated data.
- Gap: English-centric, not morphology-focused.

## 4) Phi-1 (Gunasekar et al., 2023)
- Data quality over sheer data scale.
- Gap: no Arabic morphology component.

## 5) BabyLM Challenge (Warstadt et al., 2023)
- Developmental/limited-data benchmarking.
- Gap: no Arabic Semitic morphology.

## 6) Morphological Word Embeddings (Cotterell et al.)
- Theoretical basis for compositional morphology-aware embeddings.
- Gap: pre-transformer setting, not modern decoder-only AR LMs.

## 7) ArabERT (Antoun et al., 2020)
- Core Arabic LM baseline family.
- Gap: no explicit root/pattern factorization.

## 8) Arabic probing studies ("What BERT knows about Arabic")
- Methodological references for morphology probing.
- Gap: encoder setting, not decoder-only small models.

## 9) UL2 (Tay et al., 2022)
- Multi-objective training motivation for auxiliary losses.
- Gap: no direct morphology objective.

## 10) Data-constrained LM scaling studies
- Practical guidance for limited compute/data regimes.
- Gap: limited Arabic-specific morphology framing.

## Consolidated Gap
No prior work directly evaluates explicit morphology-aware training/inputs for small (≤30M) Arabic decoder-only transformers with phased ablations.
