#!/bin/bash

python sentence_llm_evaluator.py csabstruct llama3 19269 | tee -a _logs/experiments.llama3.log
python sentence_llm_evaluator.py pubmed200k llama3 19269 | tee -a _logs/experiments.llama3.log
python sentence_llm_evaluator.py coarsediscourse llama3 19269 | tee -a _logs/experiments.llama3.log
python sentence_llm_evaluator.py dailydialog llama3 19269 | tee -a _logs/experiments.llama3.log
python sentence_llm_evaluator.py emotionlines llama3 19269 | tee -a _logs/experiments.llama3.log

python sentence_llm_evaluator.py csabstruct gemma2 19269 | tee -a _logs/experiments.gemma2.log
python sentence_llm_evaluator.py pubmed200k gemma2 19269 | tee -a _logs/experiments.gemma2.log
python sentence_llm_evaluator.py coarsediscourse gemma2 19269 | tee -a _logs/experiments.gemma2.log
python sentence_llm_evaluator.py dailydialog gemma2 19269 | tee -a _logs/experiments.gemma2.log
python sentence_llm_evaluator.py emotionlines gemma2 19269 | tee -a _logs/experiments.gemma2.log

python sentence_llm_evaluator.py csabstruct mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_evaluator.py pubmed200k mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_evaluator.py coarsediscourse mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_evaluator.py dailydialog mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_evaluator.py emotionlines mistral 19269 | tee -a _logs/experiments.mistral.log
