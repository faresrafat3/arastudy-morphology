# Factored Morphological Embeddings for Arabic Language Models: Root-Aware Representations in Small Transformers

## Abstract
Arabic's rich morphological system derives many words from a single root, while small language models typically treat surface forms independently. This project investigates whether explicit morphological signals improve small (≤30M) Arabic decoder-only language models. We evaluate two progressive approaches: (1) morphology-aware training data that exposes root-pattern families, and (2) lightweight root-aware input features. We report baseline probing, language-model quality, and generalization to unseen forms. Results placeholders are kept until experiments complete.

## 1. Introduction
Arabic is a morphologically rich language in which dozens of words can be derived from a single three-consonant root through systematic vowel patterns. This productive morphological system creates families of semantically related words (e.g., كَتَبَ "wrote," كَاتِب "writer," كِتَاب "book," مَكْتَبَة "library" — all from the root ك-ت-ب). Despite this rich structure, current Arabic language models treat each surface form as an independent token, relying on statistical co-occurrence rather than morphological relationships.

We investigate whether small Arabic language models (≤30M parameters) can benefit from explicit morphological information, addressing three questions:

1. Do standard Arabic LMs learn morphological representations implicitly? (RQ0)
2. Does morphological training data improve these representations? (RQ1)
3. Do lightweight root embeddings provide additional benefit? (RQ2)

Our analysis of a 28.6M-parameter baseline reveals [KEY FINDING from results]. Through progressive interventions — morphological training data (Phase 1) and root embedding addition (Phase 2) — we demonstrate [MAIN RESULT].

Our contributions are:
- First systematic probing study of morphological awareness in small Arabic decoder-only transformers.
- A morphological training data approach that [RESULT].
- A lightweight root embedding architecture that adds only 1.8% parameters while [RESULT].
- Layer-wise analysis revealing [LAYER FINDING], consistent with neurolinguistic evidence.
- Complete open-source pipeline including code, morphological data, and evaluation suite.

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
| تَفْعِيل | تَعْلِيم | teaching | verbal noun (II) |
| اِسْتِفْعَال | اِسْتِخْرَاج | extraction | verbal noun (X) |
| اِفْتِعَال | اِجْتِمَاع | meeting | verbal noun (VIII) |
| اِنْفِعَال | اِنْفِجَار | explosion | verbal noun (VII) |
| مَفْعَلَة | مَكْتَبَة | library | place noun |

This productive system generates dozens of related words from a single root, creating systematic families of morphologically related forms. The Arabic root inventory contains approximately 5,000-10,000 roots, of which roughly 1,500-2,000 are commonly used in modern text.

### 2.2 Challenges for Language Models

Standard subword tokenizers such as BPE (Sennrich et al., 2016) segment text based on statistical frequency rather than morphological structure. Our analysis of BPE-16K trained on Arabic Wikipedia reveals that approximately 48% of regular Arabic morpheme boundaries are broken by BPE segmentation (Section 4.1), with this rate remaining stable across vocabulary sizes (8K-32K), indicating a fundamental limitation of frequency-based tokenization for morphologically rich languages.

This misalignment between tokenization and morphological structure means that language models must discover morphological relationships implicitly from co-occurrence patterns, rather than benefiting from explicit structural information.

Additionally, most Arabic text is written without diacritical marks (tashkeel), which encode case endings and short vowels. This means that models must disambiguate morphological forms from context alone, adding to the challenge.

### 2.3 Morphological Categories

Arabic words fall into several morphological categories relevant to our analysis:

- **Regular trilateral** (~50%): Standard three-consonant roots with predictable derivation patterns.
- **Weak roots** (~15%): Roots containing semi-vowels (و، ي) that undergo phonological changes (e.g., قال from root ق-و-ل).
- **Broken plurals** (~15%): Non-concatenative plural formation (e.g., كُتُب from كِتَاب) that cannot be described as simple affixation.
- **Quadriliteral** (~10%): Four-consonant roots (e.g., تَرْجَمَ "to translate").
- **Function words** (~10%): Particles, pronouns, and prepositions without derivational roots.

## 3. Related Work
### 3.1 Arabic Language Models
Recent years have seen growing interest in Arabic-specific language models. CAMeLBERT (Inoue et al., 2021) trained multiple BERT variants on Arabic text and demonstrated through probing experiments that encoder-based models partially capture morphological information. Jais (Sengupta et al., 2023) introduced an Arabic-English bilingual LLM with Arabic-aware BPE tokenization, showing that morphological pre-segmentation improves tokenization quality. ArabERT (Antoun et al., 2020) provided early Arabic BERT models using WordPiece tokenization. SILMA (2024) introduced efficient Arabic LMs targeting edge deployment. Kuwain (2024) explored bilingual Arabic-English injection for small models. However, none of these works explicitly encode morphological structure in the model architecture or training data for small decoder-only models.

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

### 4.2 Baseline Model

Our baseline is a decoder-only transformer following the LLaMA architecture (Touvron et al., 2023). We train two model sizes to investigate the interaction between model capacity and morphological learning:

| Component | 28.6M | 10M |
|-----------|-------|------|
| Dimension | 512 | 384 |
| Layers | 6 | 4 |
| Heads | 8 | 6 |
| FFN hidden | 1,536 | 1,024 |
| Max seq len | 512 | 512 |
| Vocab size | 16,000 | 16,000 |
| Position | RoPE | RoPE |
| FFN type | SwiGLU | SwiGLU |
| Norm | RMSNorm | RMSNorm |
| Weight tying | Yes | Yes |
| **Total params** | **28.6M** | **~10M** |

Both models use Rotary Position Embeddings (RoPE; Su et al., 2021) rather than learned positional embeddings, SwiGLU activation (Shazeer, 2020), and RMSNorm (Zhang and Sennrich, 2019). Input and output embedding weights are tied.

### 4.3 Morphological Training Data

To explicitly expose the model to Arabic morphological structure, we construct root-word family lists using CAMeL Tools morphological analyzer (Obeid et al., 2020).

**Data construction.** We extract the 50,000 most frequent unique words from our Arabic Wikipedia corpus and analyze each word using CAMeL Tools to identify its root. We then group words by root, filtering to retain only roots that: (1) consist of valid Arabic characters (2-5 letters), (2) have at least 3 attested word forms in the corpus.

This yields 1,151 root families covering 13,672 word forms. Each family is formatted as a single training line:

> جذر كتب: كتب كاتب مكتوب كتاب كتابة مكتبة

**Data mixing.** Following the controlled experiment principle, we replace 5% of Wikipedia training lines with morphological root-word lists, maintaining the same total training tokens. This ensures that any improvement is attributable to data quality rather than quantity.

**Token-root coverage.** Of the 16,000 BPE tokens in our vocabulary, 6,213 (39%) can be mapped to a known Arabic root. The remaining 61% includes function words, foreign terms, and subword fragments without clear root association.

### 4.4 Root Embedding Addition

In Phase 2, we augment the model with a lightweight root embedding layer. For each input token, we look up its associated Arabic root (pre-computed using CAMeL Tools) and add the root embedding to the token embedding:

$$e_t = E_{tok}(x_t) + \alpha \cdot E_{root}(r_t)$$

where $E_{tok}$ is the standard BPE embedding, $E_{root}$ is a learned root embedding table (1,015 roots × 512 dimensions = 520K parameters), $r_t$ is the root ID for token $x_t$ (0 for unknown), and $\alpha$ is a learned scalar scaling factor initialized to 0.1.

We initialize $\alpha = 0.1$ to prevent the randomly initialized root embeddings from disrupting the pretrained token embeddings in early training. The model learns to increase $\alpha$ as root embeddings become meaningful.

This adds only 520K parameters (1.8% increase over the baseline), preserves weight tying between input and output embeddings, and requires no changes to the transformer architecture beyond the embedding layer.

Tokens without a known root (61% of vocabulary) receive only the standard BPE embedding, functioning as an automatic fallback.

### 4.5 Training Setup

**Corpus.** We train on Arabic Wikipedia (November 2023 dump), cleaned with two rounds of filtering that remove short lines (<10 words), lines with less than 70% Arabic characters, list entries, and demographic table fragments. The final corpus contains 4.2 million lines (~238 million words). We split 95/5 into training (4.03M lines) and validation (212K lines) sets.

**Tokenization.** We train a BPE tokenizer with 16,000 vocabulary using SentencePiece (Kudo and Richardson, 2018) with byte_fallback and 99.99% character coverage. The training corpus is pre-tokenized into uint16 binary arrays for efficient memory-mapped loading during training.

**Optimization.** We use AdamW (Loshchilov and Hutter, 2019) with $\beta_1=0.9$, $\beta_2=0.95$, weight decay 0.1, and gradient clipping at 1.0. Learning rate follows a cosine schedule with linear warmup (1,000 steps) from peak $3 \times 10^{-4}$ to minimum $3 \times 10^{-5}$. We use mixed precision (FP16) with gradient accumulation (effective batch size 64, sequence length 512).

**Training duration.** Based on validation loss monitoring, we find that the 28.6M baseline overfits after approximately 42K steps and the 10M baseline after approximately 8K steps. We use best-checkpoint selection based on validation loss with early stopping (patience = 5 evaluations).

We define Root Clustering Score (RCS) as:

$$
	ext{RCS} = \frac{1}{|S|} \sum_{(i,j) \in S} \cos(e_i, e_j) - \frac{1}{|D|} \sum_{(i,k) \in D} \cos(e_i, e_k)
$$

where $S$ is the set of same-root word pairs and $D$ is the set of different-root pairs.

**Evaluation metrics.** We report: (1) Root Clustering Score (RCS), measuring whether embeddings of morphologically related words are more similar than unrelated words; (2) Perplexity (PPL) on held-out validation data; (3) statistical significance via bootstrap permutation test (n=1,000). Layer-wise probing examines morphological information at each transformer layer.

## 5. Experiments and Results
### 5.1 Baseline Morphological Analysis (RQ0)

We first investigate whether standard training leads to morphological representations in our baseline models.

**Root Clustering Score.** The 28.6M baseline achieves RCS = 0.132 (intra-root similarity 0.514 vs. inter-root similarity 0.382). Bootstrap permutation testing (n=1,000) confirms this is highly significant (p < 0.001, 9.5σ above the random baseline mean of 0.000 ± 0.014).

**Layer-wise analysis.** Figure 1 reveals a striking pattern: morphological information is concentrated in the earliest layer (Layer 1: RCS = 0.169) and diminishes monotonically through later layers, becoming negative by Layer 3 (RCS = −0.011 to −0.034). This suggests the model initially represents morphological similarity through orthographic overlap but is progressively replaced by contextual representations.

This pattern is consistent with neurolinguistic evidence showing that morphological processing precedes semantic processing in human language comprehension (~150ms vs ~400ms; Rastle et al., 2004).

**Overfitting analysis.** The 28.6M model shows optimal validation loss at step 42K (val_loss = 3.678), after which validation loss increases while training loss continues to decrease, indicating overfitting. The 10M model overfits 5× faster (at step 8K), suggesting that smaller models exhaust their learning capacity more quickly and may benefit more from targeted morphological training.

**Summary.** Standard Arabic LMs partially learn morphological structure (RCS = 0.132, p < 0.001), but this knowledge is shallow (Layer 1 only) and the model shows signs of capacity exhaustion. This motivates our morphological interventions in Phase 1 and Phase 2.

### 5.2 Phase 1: Morphological Training Data (RQ1)

[TABLE 2: Baseline vs +MorphData — fill when results arrive!]

| Model | Val Loss | PPL | RCS | RCS p-value |
|-------|----------|-----|-----|-------------|
| Baseline | 3.678 | 39.6 | 0.132 | <0.001 |
| +MorphData | ??? | ??? | ??? | ??? |

[Analysis — fill when results arrive!]

### 5.3 Phase 2: Root Embedding (RQ2)

[TABLE 3: All models comparison — fill when results arrive!]

| Model | Params | Val Loss | PPL | RCS |
|-------|--------|----------|-----|-----|
| Baseline 28.6M | 28.6M | 3.678 | 39.6 | 0.132 |
| +MorphData (P1) | 28.6M | ??? | ??? | ??? |
| +RootEmb (P2) | 29.2M | ??? | ??? | ??? |
| Baseline 10M | ~10M | 4.238 | ??? | ??? |

[TABLE 4: Ablation — fill when results arrive!]

### 5.4 Generalization and Layer Analysis (RQ3, RQ4)

[FIGURE 2: Layer-wise RCS comparison across models!]
[FIGURE 3: Unseen word generalization!]
[Analysis — fill when results arrive!]

## 6. Discussion

### 6A. Discussion (If Phase 1/2 improve RCS clearly)

#### 6A.1 Morphological Awareness in Small Arabic LMs

Our results reveal that small Arabic language models exhibit a nuanced relationship with morphological structure. The baseline model (28.6M parameters) achieves an RCS of 0.132 (p < 0.001), demonstrating that standard training on Arabic text leads to partial morphological awareness even without explicit supervision. However, this awareness is concentrated in the earliest transformer layer (Layer 1: RCS = 0.169) and diminishes in deeper layers, suggesting that morphological similarity is initially captured through orthographic overlap but is progressively overwritten by contextual representations.

Phase 1 (morphological training data) increases the RCS to [X], representing a [Y%] improvement over the baseline. Phase 2 (root embeddings) further improves the RCS to [Z], demonstrating that lightweight architectural modifications can complement data-level interventions.

#### 6A.2 The Layer-wise Dissociation

The monotonic decrease in RCS from Layer 1 to Layer 6 parallels findings in neurolinguistics, where morphological processing (~150ms) precedes semantic processing (~400ms) in human language comprehension (Rastle et al., 2004). Our models appear to follow a similar hierarchy: early layers capture form-level (morphological) similarity, while later layers prioritize meaning-level (semantic) representations.

[If Phase 1/2 changes the layer pattern:] Notably, morphological training [data/root embeddings] alters this pattern, [preserving/strengthening] morphological information in deeper layers. This suggests that explicit morphological signals can help the model maintain awareness of word structure alongside contextual meaning.

#### 6A.3 Capacity and Overfitting

The 28.6M baseline overfits after 42K steps, while the 10M baseline overfits after only 8K steps (5× faster). This differential suggests that smaller models exhaust their capacity to learn from raw text more quickly and may benefit disproportionately from structured morphological input.

[If 10M benefits more from morphological approach:] Indeed, the 10M model shows a [larger/similar] relative improvement from morphological training ([X%] vs [Y%] for the 28.6M model), supporting the hypothesis that morphological inductive bias is particularly valuable for capacity-constrained models.

#### 6A.4 Limitations

Several limitations should be noted:

1. **Orthographic confound.** The observed Layer 1 RCS may partly reflect shared character sequences rather than true morphological awareness. Words sharing a root necessarily share consonants, making it difficult to fully disentangle orthographic from morphological similarity.

2. **Root analysis accuracy.** Our morphological analysis relies on CAMeL Tools, which achieves approximately 90% accuracy. The 10% error rate introduces noise in both the morphological training data (Phase 1) and root embeddings (Phase 2).

3. **Single language.** Our experiments focus exclusively on Arabic. While the approach should generalize to other Semitic languages with similar root-pattern systems (e.g., Hebrew), this remains to be verified.

4. **Model scale.** We study models at 10M and 28.6M parameters. The benefits of explicit morphological encoding may diminish at larger scales where models have sufficient capacity to discover morphological patterns independently.

5. **Correlation vs. causation.** Our probing methodology measures correlation between learned representations and morphological structure, not causal understanding of morphology by the model.

#### 6A.5 Future Work

Several directions emerge from this work:

- **Cross-lingual transfer:** Applying our approach to Hebrew and other morphologically rich languages.
- **Factored embeddings:** Replacing additive root embeddings with factored representations (word = transform × root) for stronger structural bias.
- **Morphological attention bias:** Modifying attention to explicitly connect morphologically related tokens.
- **Scaling analysis:** Investigating how morphological benefits interact with model size up to 1B+ parameters.

### 6B. Discussion (If Phase 1/2 gains are limited)

#### 6B.1 Implicit Morphological Learning

Our central finding is that a 28.6M-parameter Arabic language model, trained with standard next-token prediction on Wikipedia, achieves statistically significant morphological awareness (RCS = 0.132, p < 0.001, 9.5σ). This suggests that Arabic morphological structure is partially discoverable through distributional patterns alone, consistent with connectionist theories of implicit morphological learning.

#### 6B.2 The Limits of Explicit Intervention

Despite the statistical significance of baseline morphological awareness, our explicit interventions show [limited/modest] additional benefit:

- Phase 1 (morphological training data): RCS = [X] ([Y%] change from baseline 0.132)
- Phase 2 (root embeddings): RCS = [Z] ([W%] change from baseline)

This suggests that the model's implicit learning may already capture the morphological information that our explicit signals provide. The [small/modest] improvement indicates that more fundamental architectural changes (e.g., factored embeddings, morphological attention) may be needed to substantially improve morphological representations beyond what standard training achieves.

#### 6B.3 Why Explicit Morphology May Not Help More

Several factors may explain the limited improvement:

1. **Distributional sufficiency:** Arabic morphological families co-occur frequently in natural text, providing rich distributional cues that standard training captures.

2. **BPE alignment:** Despite breaking 48% of morpheme boundaries, BPE preserves enough sub-morphemic information for the model to reconstruct morphological relationships.

3. **Representation capacity:** The model may allocate its limited capacity to contextual rather than morphological representations, regardless of training signal, because contextual information is more useful for next-token prediction.

#### 6B.4 Limitations and Future Work

The same limitations in Section 6A.4 apply (orthographic confound, analyzer noise, single-language scope, model-size range, and correlation-vs-causation constraints). Future work should prioritize stronger structural inductive biases (factored embeddings, morphology-aware attention) and multilingual validation.

## 7. Conclusion

### 7A. Conclusion (If Phase 1/2 improve RCS clearly)

We investigated morphological awareness in small Arabic language models through three progressive interventions. Our baseline analysis reveals that a 28.6M-parameter model achieves partial morphological awareness (RCS = 0.132, p < 0.001), concentrated in early layers and consistent with neurolinguistic evidence of early morphological processing.

Adding morphological training data (Phase 1) improves the RCS to [X] ([Y%] improvement), while root embedding addition (Phase 2) further improves it to [Z] ([W%] improvement), demonstrating that both data-level and architectural interventions can enhance morphological representations in small Arabic models.

Our tokenizer analysis shows that BPE breaks 48% of regular Arabic morpheme boundaries regardless of vocabulary size, motivating morphology-aware approaches. The layer-wise probing methodology introduced here provides a tool for understanding how morphological information flows through transformer layers.

We release our code, morphological training data, root embedding implementation, and evaluation suite to facilitate future work on morphologically-aware Arabic language models.

### 7B. Conclusion (If gains are modest)

We present the first systematic study of morphological awareness in small Arabic decoder-only transformers. Our key finding is that a 28.6M-parameter model trained on Arabic Wikipedia achieves statistically significant morphological awareness (RCS = 0.132, p < 0.001) without explicit morphological supervision. This awareness is concentrated in early layers, paralleling the hierarchy of morphological and semantic processing observed in human language comprehension.

Our morphological interventions (training data and root embeddings) show [modest] improvements, suggesting that standard distributional learning captures much of the morphological structure available in Arabic text. This finding has implications for Arabic NLP: rather than adding morphological features, researchers may benefit from focusing on other bottlenecks in small model performance.

We release our complete experimental pipeline, including probing methodology, morphological training data, and analysis tools.

## Evidence Checklist
- Tables: baseline, data approach, architecture approach, ablation.
- Figures: root clustering, unseen forms, layer-wise probing.
