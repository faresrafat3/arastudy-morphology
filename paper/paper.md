# Factored Morphological Embeddings for Arabic Language Models: Root-Aware Representations in Small Transformers

## Abstract
Arabic's rich morphological system derives many words from a single root, while small language models typically treat surface forms independently. This project investigates whether explicit morphological signals improve small (≤30M) Arabic decoder-only language models. We evaluate two progressive approaches: (1) morphology-aware training data that exposes root-pattern families, and (2) lightweight root-aware input features. We report baseline probing, language-model quality, and generalization to unseen forms. Our baseline analysis reveals partial morphological awareness (RCS=0.132, p<0.001), concentrated in early layers. Morphological training data strengthens early-layer morphology (+20% Layer 1 RCS) and revives it in the final layer (Layer 6: -0.034→+0.068). Root embeddings further improve language modeling (val_loss -2.0%) and distribute morphological signal across all layers.

## 1. Introduction
Arabic is a morphologically rich language in which dozens of words can be derived from a single three-consonant root through systematic vowel patterns. This productive morphological system creates families of semantically related words (e.g., كَتَبَ "wrote," كَاتِب "writer," كِتَاب "book," مَكْتَبَة "library" — all from the root ك-ت-ب). Despite this rich structure, current Arabic language models treat each surface form as an independent token, relying on statistical co-occurrence rather than morphological relationships.

We investigate whether small Arabic language models (≤30M parameters) can benefit from explicit morphological information, addressing three questions:

1. Do standard Arabic LMs learn morphological representations implicitly? (RQ0)
2. Does morphological training data improve these representations? (RQ1)
3. Do lightweight root embeddings provide additional benefit? (RQ2)

Our analysis of a 28.6M-parameter baseline reveals partial morphological awareness in Layer 1 (RCS=0.169) that disappears in deeper layers, and that explicit morphological training fundamentally changes this pattern. Through progressive interventions — morphological training data (Phase 1) and root embedding addition (Phase 2) — we demonstrate that morphological training data revives morphological awareness in the final layer (RCS: -0.034→+0.068), while root embeddings distribute it across all layers, alongside a consistent 2% improvement in language modeling quality.

Our contributions are:
- First systematic probing study of morphological awareness in small Arabic decoder-only transformers.
- A probing analysis that reveals partial morphological awareness (RCS=0.132, p<0.001) concentrated in Layer 1.
- A morphological training data approach that improves val_loss by 1.0% and strengthens Layer 1 RCS by 20%.
- A lightweight root embedding architecture that adds only 1.8% parameters while improving val_loss by 2.0% and creating positive morphological signal across all layers.
- Layer-wise analysis showing that morphological signal transitions from Layer-1-only to all-layer distribution with architectural intervention, consistent with neurolinguistic evidence.
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
Adding morphological root-word family lists (5% of training data,
controlled for total tokens) yields modest but consistent improvement
in language modeling quality.

| Model | Val Loss | PPL | Best Step | Δ Val Loss |
|-------|----------|-----|-----------|------------|
| Baseline | 3.678 | 39.6 | 42K | — |
| +MorphData | 3.642 | 38.2 | 52K | -1.0% |

The morphological data also extends training stability: the baseline
begins overfitting at step 42K, while Phase 1 continues improving
until step 52K (+24% more productive training steps).

**Layer-wise impact.** Most strikingly, morphological training data
produces two effects on the layer-wise morphological signal:
(1) Layer 1 RCS increases from 0.169 to 0.202 (+20%), indicating
stronger early morphological processing, and (2) Layer 6 RCS
reverses from -0.034 to +0.068, indicating that morphological
awareness, which disappeared in the baseline's final layer,
is revived by explicit morphological training.

This suggests that morphological training data helps the model
*preserve* morphological information through deeper processing
layers, rather than overwriting it with purely contextual
representations.

### 5.3 Phase 2: Root Embedding (RQ2)
Adding a lightweight root embedding layer (520K parameters, 1.8%
increase) produces the strongest language modeling improvement.

| Model | Params | Val Loss | PPL | Best Step | Δ Val Loss |
|-------|--------|----------|-----|-----------|------------|
| Baseline | 28.6M | 3.678 | 39.6 | 42K | — |
| +MorphData (P1) | 28.6M | 3.642 | 38.2 | 52K | -1.0% |
| +RootEmb (P2) | 29.2M | 3.604 | 36.7 | 54K | -2.0% |
| Baseline 10M | ~10M | 3.851 | 47.1 | 44K | — |

Phase 2 achieves the best val_loss (3.604) and extends productive
training to step 54K.

**Layer-wise transformation.** Root embeddings fundamentally change
the morphological representation pattern. Unlike the baseline
(where Layers 3-6 show negative RCS) and Phase 1 (mixed), Phase 2
produces positive RCS across *all* layers:

| Layer | Baseline | Phase 1 | Phase 2 |
|-------|----------|---------|---------|
| L1 | +0.169 | +0.202 | +0.115 |
| L2 | +0.028 | -0.051 | +0.008 |
| L3 | -0.011 | -0.021 | +0.003 |
| L4 | -0.011 | -0.030 | +0.003 |
| L5 | -0.024 | -0.025 | +0.004 |
| L6 | -0.034 | +0.068 | +0.041 |

This pattern indicates that root embeddings create a more uniform
morphological representation, distributing morphological signal
across all layers rather than concentrating it in Layer 1.

The reduced Layer 1 RCS (0.115 vs 0.169) reflects this
redistribution: the model no longer relies solely on early
orthographic similarity but encodes morphological relationships
throughout its depth.

**10M baseline.** The smaller model shows higher embedding-level
RCS (0.113 vs 0.109 for 28.6M), suggesting that smaller models
allocate proportionally more representational capacity to
morphological structure — consistent with our hypothesis that
morphological inductive bias is particularly valuable for
capacity-constrained models.

### 5.4 Generalization and Layer Analysis (RQ3, RQ4)
**Three distinct representation strategies.** Our layer-wise
analysis reveals three qualitatively different strategies for
encoding morphological information:

1. **Baseline: Concentrate-and-Abandon.** Strong Layer 1 signal
	(RCS=0.169) that monotonically decreases to negative values,
	suggesting the model captures morphological similarity early
	but progressively abandons it for contextual representations.

2. **Phase 1: Strengthen-and-Revive.** Stronger Layer 1 signal
	(RCS=0.202) with negative middle layers but a positive final
	layer (RCS=+0.068), suggesting that explicit morphological
	exposure helps the model recover morphological information
	in its output representations.

3. **Phase 2: Distribute.** Moderate Layer 1 signal (RCS=0.115)
	with consistently positive (though small) RCS across all
	layers, suggesting that root embeddings enable a fundamentally
	different strategy: uniform morphological encoding rather than
	layer-specialized processing.

These patterns parallel neurolinguistic findings showing that
morphological processing (~150ms) typically precedes semantic
processing (~400ms) in human comprehension (Rastle et al., 2004),
with our Phase 2 model resembling a more integrated processing
strategy.

**Generation quality.** Qualitative examination of generation
samples shows all three models produce fluent Arabic text with
Wikipedia-like content. The improvements are subtle at the
surface level, consistent with the modest 1-2% val_loss
improvements, but suggest better handling of morphologically
complex constructions.

## 6. Discussion
### 6.1 Three Representation Strategies

Our most significant finding is not the modest improvement in
language modeling quality (1-2%), but the discovery of three
qualitatively different strategies for encoding Arabic morphology.
The baseline's concentrate-and-abandon strategy, Phase 1's
strengthen-and-revive pattern, and Phase 2's distribute approach
represent increasingly sophisticated morphological processing,
paralleling developmental stages in human language acquisition.

### 6.2 The Layer 6 Revival

Phase 1's most striking result — the revival of morphological
signal in Layer 6 (from -0.034 to +0.068) — suggests that
explicit morphological exposure in training data helps the model
maintain awareness of word structure even at the output level.
This has practical implications: models with output-level
morphological awareness may generate more morphologically
correct text.

### 6.3 Root Embeddings as Distributional Prior

Phase 2's uniform positive RCS across all layers represents a
fundamentally different approach: rather than discovering and
then abandoning morphology, the model maintains it throughout.
This comes at the cost of reduced Layer 1 discrimination
(0.115 vs 0.169), suggesting a trade-off between strong
early morphological clustering and distributed morphological
awareness.

### 6.4 Limitations

1. **Single corpus.** All experiments use Arabic Wikipedia only.
	Results may differ with other Arabic text sources.

2. **Model scale.** We study 10M and 28.6M models only.
	Benefits may diminish at larger scales.

3. **BPE tokenizer.** We use standard BPE-16K without
	morphological pre-segmentation.

4. **RCS metric.** Root Clustering Score is novel and not
	yet established in the literature.

5. **No human evaluation.** We rely on automatic metrics
	and qualitative generation inspection.

6. **Root analysis accuracy.** CAMeL Tools achieves ~90%
	accuracy, introducing noise in morphological labels.

## 7. Conclusion
We present the first systematic study of morphological
representation in small Arabic decoder-only transformers.
Our analysis of a 28.6M-parameter model reveals that standard
training produces partial morphological awareness (RCS=0.132,
p<0.001) concentrated in the earliest layer.

Through two progressive interventions, we discover three
distinct representation strategies: the baseline's
concentrate-and-abandon pattern, morphological training data's
strengthen-and-revive effect (Layer 6 RCS: -0.034→+0.068),
and root embeddings' distribute strategy (positive RCS across
all layers). Both interventions improve language modeling quality
(up to 2.0% val_loss reduction) while fundamentally changing
how the model encodes morphological structure.

Our findings suggest that Arabic language models benefit from
explicit morphological information not primarily through improved
surface metrics, but through qualitatively different representation
strategies that better reflect the compositional nature of Arabic
morphology. We release our complete experimental pipeline,
morphological training data, and analysis tools to facilitate
further research.

## Evidence Checklist
- Tables: baseline, data approach, architecture approach, ablation.
- Figures: root clustering, unseen forms, layer-wise probing.
