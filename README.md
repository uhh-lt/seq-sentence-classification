# Sequential Sentence Classification

This repository contains the benchmark experiments for the paper Semi-automatic Sequential Sentence Classification in the Discourse Analysis Tool Suite.
The Discourse Analysis Tool Suite can be found here: https://github.com/uhh-lt/dats

## Goal

We propose a three-phase strategy to support users of DATS with the best sentence annotation suggestions as possible.
Depending on the number of annotations, we employ a different strategy to predict suggestions.

1. Phase 1: Zero-shot prompting of LLMs
2. Phase 2: Few-shot prompting of LLMs
3. Phase 3: Training of a Sequential Sentence Tagger

## Models

We evaluate the following models:

- LLMs: Llama, Mistral Nemo, Gemma 2
- Sentence Taggers with this architecture: Sentence Transformer embeddings + BiLSTM + CRF

## Datasets

We created a "benchmark" consisting of five datasets, but it is planned to extend the benchmark:

- [Coarse Discourse](./datasets/coarsediscourse/)
- [CSAbstruct](./datasets/csabstruct/)
- [Daily Dialog](./datasets/daily_dialog/)
- [Emotion Lines](./datasets/emotion_lines/)
- [Pubmed200k](./datasets/pubmed200k/)

## Results
| Dataset    	|      	| CSAbstruct 	|       	| Pubmed200k 	|       	| CoarseDiscourse 	|       	| EmotionLines 	|       	| DailyDialog 	|       	|
|------------	|------	|------------	|-------	|------------	|-------	|-----------------	|-------	|--------------	|-------	|-------------	|-------	|
| Model      	| Shot 	| F1         	| Acc   	| F1         	| Acc   	| F1              	| Acc   	| F1           	| Acc   	| F1          	| Acc   	|
| Llama 3.1  	| 0    	| 35.68      	| 49.67 	| 33.91      	| 60.27 	| 27.64           	| 23.59 	| 23.19        	| 28.61 	| 26.16       	| 30.60 	|
| Gemma 2    	| 0    	| 40.02      	| 55.08 	| 52.88      	| 73.32 	| 33.28           	| 32.36 	| 33.26        	| 46.92 	| 31.32       	| 40.83 	|
| Mistral    	| 0    	| 39.39      	| 56.63 	| 45.85      	| 70.98 	| 31.64           	| 30.73 	| 18.68        	| 27.64 	| 26.25       	| 31.14 	|
| Llama 3.1  	| 2    	| 37.71      	| 51.96 	| 48.43      	| 71.54 	| 27.54           	| 23.95 	| 23.53        	| 28.94 	| 27.86       	| 33.39 	|
| Gemma 2    	| 2    	| 45.82      	| 60.79 	| 56.93      	| 72.02 	| 18.21           	| 11.77 	| 29.86        	| 33.06 	| 30.85       	| 37.51 	|
| Mistral    	| 2    	| 41.35      	| 57.60 	| 53.95      	| 76.06 	| 31.94           	| 31.47 	| 22.49        	| 28.61 	| 27.46       	| 36.59 	|
| Llama 3.1  	| 4    	| 35.09      	| 50.63 	| 49.22      	| 69.93 	| 26.79           	| 23.11 	| 22.19        	| 27.98 	| 25.83       	| 30.02 	|
| Gemma 2    	| 4    	| 45.78      	| 57.89 	| 50.43      	| 51.93 	| 13.59           	| 08.16 	| 26.85        	| 29.55 	| 22.60       	| 26.44 	|
| Mistral    	| 4    	| 44.57      	| 60.27 	| 54.04      	| 76.95 	| 32.85           	| 31.96 	| 23.99        	| 32.61 	| 27.80       	| 37.86 	|
| allmpnetv2 	| All  	| 37.43      	| 61.23 	| 72.89      	| 85.78 	| 29.15           	| 41.48 	| 26.06        	| 47.14 	| 22.03       	| 51.81 	|
| NV-Embed   	| All  	| 55.65      	| 71.98 	| 81.72      	| 90.20 	| 42.40           	| 51.05 	| 34.86        	| 52.89 	| 31.77       	| 57.67 	|
| SOTA       	| All  	| 83.1       	| --    	| 93.1       	| --    	| 84.0            	| --    	| 69.90        	| 68.7  	| 64.2        	| --    	|
