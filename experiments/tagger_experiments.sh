#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python tagger_precomputed_experiment.py csabstruct query_embedding
CUDA_VISIBLE_DEVICES=1 python tagger_precomputed_experiment.py csabstruct passage_embedding

CUDA_VISIBLE_DEVICES=1 python tagger_precomputed_experiment.py dailydialog query_embedding
CUDA_VISIBLE_DEVICES=1 python tagger_precomputed_experiment.py dailydialog passage_embedding

CUDA_VISIBLE_DEVICES=1 python tagger_precomputed_experiment.py emotionlines query_embedding
CUDA_VISIBLE_DEVICES=1 python tagger_precomputed_experiment.py emotionlines passage_embedding

CUDA_VISIBLE_DEVICES=1 python tagger_precomputed_experiment.py pubmed200k query_embedding
CUDA_VISIBLE_DEVICES=1 python tagger_precomputed_experiment.py pubmed200k passage_embedding

CUDA_VISIBLE_DEVICES=1 python tagger_precomputed_experiment.py coarsediscourse query_embedding
CUDA_VISIBLE_DEVICES=1 python tagger_precomputed_experiment.py coarsediscourse passage_embedding