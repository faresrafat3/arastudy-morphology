# AraStudy: Factored Morphological Embeddings for Arabic LMs

AraStudy is a research-oriented Arabic language modeling project focused on studying whether explicit morphological structure (roots, patterns, and related forms) can improve small decoder-only language models. The project starts with strong baselines and probing methodology, then incrementally evaluates morphology-aware data and architecture choices.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Verify CAMeL tools import:
   - `python -c "from camel_tools.morphology.analyzer import Analyzer; print('CAMeL OK!')"`
4. Start with experiment config:
   - `configs/experiment/exp001_baseline.yaml`

## Notes

- This repository uses a configuration-first workflow.
- Baseline-first evaluation is recommended before adding architectural complexity.
