# Sequential Sentence Classification

This repository contains experiments for (few-shot) sequential sentence classification.

Considered models are:

1. small variants of popular, open-source LLM families (LLama, Mistral, Gemma)
1. sequence tagger model: embedding layer > LSTM > CRF

## Limitations

- Only consider one label per sentence vs. in practice, mulitple labels and overlapping annotations are possisble
- Only consider tasks that are available in nlp research## Datasets used are from various domains
