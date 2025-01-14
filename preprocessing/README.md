# Preprocessing

## Precompute Embeddings
embedd_datasets.py uses the latest NVembedd model to compute vector representations the sentences of all datasets.
These pre-computed embeddings are used by the sequence tagger model. 
By pre-computing embeddings, we can save time during training and inference.

## Fewshot Sampling
In fewshot scenarios, you typically have a set number of classes N and a set number of labeled examples k per class. 
For fewshot experiments, we consequently need to find a fitting set of examples.
We find the least number of samples that satisfy the following condition:
For every class there must be at least k, but at most 2*k sentences.
