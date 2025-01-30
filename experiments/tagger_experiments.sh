#!/bin/bash

# NVEMBED precomputed embeddings
CUDA_VISIBLE_DEVICES=1 python tagger_experiment precomputed-embeddings csabstruct
CUDA_VISIBLE_DEVICES=1 python tagger_experiment precomputed-embeddings dailydialog
CUDA_VISIBLE_DEVICES=1 python tagger_experiment precomputed-embeddings emotionlines
CUDA_VISIBLE_DEVICES=1 python tagger_experiment precomputed-embeddings pubmed200k
CUDA_VISIBLE_DEVICES=1 python tagger_experiment precomputed-embeddings coarsediscourse

# all-mpnet-base-v2, frozen
CUDA_VISIBLE_DEVICES=1 python tagger_experiment compute-embeddings csabstruct sentence-transformers/all-mpnet-base-v2 True
CUDA_VISIBLE_DEVICES=1 python tagger_experiment compute-embeddings dailydialog sentence-transformers/all-mpnet-base-v2 True
CUDA_VISIBLE_DEVICES=1 python tagger_experiment compute-embeddings emotionlines sentence-transformers/all-mpnet-base-v2 True
CUDA_VISIBLE_DEVICES=1 python tagger_experiment compute-embeddings pubmed200k sentence-transformers/all-mpnet-base-v2 True
CUDA_VISIBLE_DEVICES=1 python tagger_experiment compute-embeddings coarsediscourse sentence-transformers/all-mpnet-base-v2 True

# all-mpnet-base-v2
CUDA_VISIBLE_DEVICES=1 python tagger_experiment compute-embeddings csabstruct sentence-transformers/all-mpnet-base-v2 False
CUDA_VISIBLE_DEVICES=1 python tagger_experiment compute-embeddings dailydialog sentence-transformers/all-mpnet-base-v2 False
CUDA_VISIBLE_DEVICES=1 python tagger_experiment compute-embeddings emotionlines sentence-transformers/all-mpnet-base-v2 False
CUDA_VISIBLE_DEVICES=1 python tagger_experiment compute-embeddings pubmed200k sentence-transformers/all-mpnet-base-v2 False
CUDA_VISIBLE_DEVICES=1 python tagger_experiment compute-embeddings coarsediscourse sentence-transformers/all-mpnet-base-v2 False
