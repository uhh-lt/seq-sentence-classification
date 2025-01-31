#!/bin/bash

# NVEMBED precomputed embeddings
CUDA_VISIBLE_DEVICES=1 python tagger_experiment.py precomputed-embeddings csabstruct
CUDA_VISIBLE_DEVICES=1 python tagger_experiment.py precomputed-embeddings dailydialog
CUDA_VISIBLE_DEVICES=1 python tagger_experiment.py precomputed-embeddings emotionlines
CUDA_VISIBLE_DEVICES=1 python tagger_experiment.py precomputed-embeddings pubmed200k
CUDA_VISIBLE_DEVICES=1 python tagger_experiment.py precomputed-embeddings coarsediscourse

# all-mpnet-base-v2, frozen
CUDA_VISIBLE_DEVICES=1 python tagger_experiment.py compute-embeddings csabstruct sentence-transformers/all-mpnet-base-v2 True
CUDA_VISIBLE_DEVICES=1 python tagger_experiment.py compute-embeddings dailydialog sentence-transformers/all-mpnet-base-v2 True
CUDA_VISIBLE_DEVICES=1 python tagger_experiment.py compute-embeddings emotionlines sentence-transformers/all-mpnet-base-v2 True
CUDA_VISIBLE_DEVICES=1 python tagger_experiment.py compute-embeddings pubmed200k sentence-transformers/all-mpnet-base-v2 True
CUDA_VISIBLE_DEVICES=1 python tagger_experiment.py compute-embeddings coarsediscourse sentence-transformers/all-mpnet-base-v2 True

# all-mpnet-base-v2
CUDA_VISIBLE_DEVICES=1 python tagger_experiment.py compute-embeddings csabstruct sentence-transformers/all-mpnet-base-v2 False
CUDA_VISIBLE_DEVICES=1 python tagger_experiment.py compute-embeddings dailydialog sentence-transformers/all-mpnet-base-v2 False
CUDA_VISIBLE_DEVICES=1 python tagger_experiment.py compute-embeddings emotionlines sentence-transformers/all-mpnet-base-v2 False
CUDA_VISIBLE_DEVICES=1 python tagger_experiment.py compute-embeddings pubmed200k sentence-transformers/all-mpnet-base-v2 False
CUDA_VISIBLE_DEVICES=1 python tagger_experiment.py compute-embeddings coarsediscourse sentence-transformers/all-mpnet-base-v2 False
