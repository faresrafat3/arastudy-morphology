# Factored Morphological Embeddings for Arabic Language Models: Root-Aware Representations in Small Transformers

## Abstract
Arabic's rich morphological system derives many words from a single root, while small language models typically treat surface forms independently. This project investigates whether explicit morphological signals improve small (≤30M) Arabic decoder-only language models. We evaluate two progressive approaches: (1) morphology-aware training data that exposes root-pattern families, and (2) lightweight root-aware input features. We report baseline probing, language-model quality, and generalization to unseen forms. Results placeholders are kept until experiments complete.

## 1. Introduction
- Problem statement: Arabic morphology is productive and may be under-modeled by standard subword training.
- Scope: small decoder-only models (10M and 30M) under constrained compute.
- Research questions (RQ0-RQ4) and expected evidence.
- Contributions (to be finalized after results).

## 2. Background: Arabic Morphology
Arabic employs a rich morphological system centered on consonantal roots, typically consisting of three consonants that carry core semantic meaning. These roots combine with vowel patterns (أوزان) to produce diverse word forms with systematic relationships.

### 2.1 The Root-Pattern System

From the root k-t-b (كتب, "writing"), Arabic derives:

| Pattern | Form | Meaning | Gloss |
|---------|------|---------|-------|
| فَعَلَ | كَتَبَ | he wrote | past verb |
| فَاعِل | كَاتِب | writer | active participle |
| مَفْعُول | مَكْتُوب | written | passive participle |
| فِعَال | كِتَاب | book | noun |
| فِعَالَة | كِتَابَة | writing | verbal noun |
| مَفْعَلَة | مَكْتَبَة | library | place noun |

This productive system generates dozens of related words from a single root, creating systematic families of morphologically related forms. The Arabic root inventory contains approximately 5,000-10,000 roots, of which roughly 1,500-2,000 are commonly used in modern text.

### 2.2 Challenges for Language Models

Standard subword tokenizers such as BPE (Sennrich et al., 2016) segment text based on statistical frequency rather than morphological structure. Our analysis of BPE-16K trained on Arabic Wikipedia reveals that approximately 48% of regular Arabic morpheme boundaries are broken by BPE segmentation (Section 4.1), with this rate remaining stable across vocabulary sizes (8K-32K), indicating a fundamental limitation of frequency-based tokenization for morphologically rich languages.

This misalignment between tokenization and morphological structure means that language models must discover morphological relationships implicitly from co-occurrence patterns, rather than benefiting from explicit structural information.

### 2.3 Morphological Categories

Arabic words fall into several morphological categories relevant to our analysis:

- **Regular trilateral** (~50%): Standard three-consonant roots with predictable derivation patterns.
- **Weak roots** (~15%): Roots containing semi-vowels (و، ي) that undergo phonological changes (e.g., قال from root ق-و-ل).
- **Broken plurals** (~15%): Non-concatenative plural formation (e.g., كُتُب from كِتَاب) that cannot be described as simple affixation.
- **Quadriliteral** (~10%): Four-consonant roots (e.g., تَرْجَمَ "to translate").
- **Function words** (~10%): Particles, pronouns, and prepositions without derivational roots.

## 3. Related Work
### 3.1 Arabic Language Models
Recent years have seen growing interest in Arabic-specific language models. CAMeLBERT (Inoue et al., 2021) trained multiple BERT variants on Arabic text and demonstrated through probing experiments that encoder-based models partially capture morphological information. Jais (Sengupta et al., 2023) introduced an Arabic-English bilingual LLM with Arabic-aware BPE tokenization, showing that morphological pre-segmentation improves tokenization quality. ArabERT (Antoun et al., 2020) provided early Arabic BERT models using WordPiece tokenization. However, none of these works explicitly encode morphological structure in the model architecture or training data for small decoder-only models.

### 3.2 Morphological NLP
Cotterell and Schütze (2018) explored morphology-aware word embeddings in the pre-transformer era, demonstrating that morphological information improves static word representations. More recently, probing studies have shown that large pretrained models implicitly learn morphological features (Edmiston, 2020), though the extent of this learning in small models remains understudied, particularly for Arabic.

### 3.3 Efficient Small Language Models
TinyStories (Eldan and Li, 2023) demonstrated that models as small as 33M parameters can generate coherent text when trained on carefully controlled data. Phi-1 (Gunasekar et al., 2023) showed that data quality surpasses data quantity for training efficient models. The BabyLM Challenge (Warstadt et al., 2023) explored cognitively-inspired training with limited data. These works motivate our approach of improving small Arabic models through targeted morphological training rather than scaling.

### 3.4 Gap

To our knowledge, no prior work has systematically investigated (1) the extent to which small Arabic decoder-only transformers learn morphological structure, (2) whether explicit morphological training data improves this learning, or (3) whether lightweight architectural modifications (root embeddings) provide additional benefit. Our work addresses all three questions.

## 4. Method
### 4.1 Tokenizer Analysis

We first analyze how BPE tokenization interacts with Arabic morphological structure. We trained BPE tokenizers with vocabulary sizes of 8K, 16K, and 32K on our Arabic Wikipedia corpus and evaluated morpheme boundary preservation on 100 Arabic words spanning five morphological categories (regular trilateral, weak roots, broken plurals, quadriliteral roots, and function words).

| Category | 8K | 16K | 32K |
|----------|-----|------|------|
| Regular (50) | 52% | 52% | 52% |
| Weak (15) | 80% | 80% | 80% |
| Broken plural (15) | 100% | 100% | 100% |
| Quadriliteral (10) | 70% | 60% | 60% |
| Function (10) | 100% | 100% | 100% |
| **Overall (100)** | **70%** | **69%** | **69%** |

The regular trilateral category—the most productive morphological class—shows only 52% boundary preservation across all vocabulary sizes. Notably, words with the prefix م- (marking passive participles and place nouns) are systematically mis-segmented, with BPE merging the prefix with root consonants (e.g., مكتوب → مكت+وب instead of م+كتوب). This rate is stable across vocabulary sizes, indicating a structural limitation of BPE rather than a configuration issue.

Based on this analysis, we select BPE-16K as our tokenizer, balancing vocabulary coverage with sequence length efficiency.

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
