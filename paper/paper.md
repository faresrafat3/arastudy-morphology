# Factored Morphological Embeddings for Arabic Language Models: Root-Aware Representations in Small Transformers

## Abstract
Arabic's rich morphological system derives many words from a single root, while small language models typically treat surface forms independently. This project investigates whether explicit morphological signals improve small (≤30M) Arabic decoder-only language models. We evaluate two progressive approaches: (1) morphology-aware training data that exposes root-pattern families, and (2) lightweight root-aware input features. We report baseline probing, language-model quality, and generalization to unseen forms. Results placeholders are kept until experiments complete.

## 1. Introduction
- Problem statement: Arabic morphology is productive and may be under-modeled by standard subword training.
- Scope: small decoder-only models (10M and 30M) under constrained compute.
- Research questions (RQ0-RQ4) and expected evidence.
- Contributions (to be finalized after results).

## 2. Background: Arabic Morphology
- Root-pattern system overview.
- Regular derivation examples.
- Weak roots, broken plurals, and quadriliteral roots.
- Why these phenomena matter for representation learning.

## 3. Related Work
### 3.1 Arabic Language Models
- CAMeLBERT, Jais, ArabERT.

### 3.2 Morphological NLP
- Morphology-aware embeddings and probing work.

### 3.3 Efficient Small Language Models
- TinyStories, Phi-1, BabyLM, and data-constrained LM studies.

## 4. Method
### 4.1 Evaluation Methodology (Pre-RQ)
- Metrics overview table.

### 4.2 Baseline Model (RQ0)
- Decoder-only transformer config.

### 4.3 Morphological Training Data (RQ1)
- Data construction and mixing strategy.

### 4.4 Root Embedding Addition (RQ2)
- Equation: $e_t = E_{tok}(x_t) + E_{root}(r_t)$

### 4.5 Training Setup
- Hyperparameters and reproducibility details.

## 5. Experiments & Results
### 5.1 Baseline Analysis (RQ0)
- Table 1 + Figure 1 placeholders.

### 5.2 Morphological Data Approach (RQ1)
- Table 2 placeholder.

### 5.3 Root Embedding Approach (RQ2)
- Table 3 + Table 4 placeholders.

### 5.4 Unseen Form Generalization (RQ3)
- Figure 2 placeholder.

### 5.5 Layer-wise Probing (RQ4)
- Figure 3 placeholder.

## 6. Discussion
- Interpretation of gains and failure cases.
- Limitations and external validity.

## 7. Conclusion
- Summary and next steps.

## Evidence Checklist
- Tables: baseline, data approach, architecture approach, ablation.
- Figures: root clustering, unseen forms, layer-wise probing.
