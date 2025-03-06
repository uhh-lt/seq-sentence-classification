#!/bin/bash

python sentence_llm_evaluator.py csabstruct --is-fewshot --num-fewshot-examples 2 --anno-guide-in-user-prompt gemma2 19269 | tee -a _logs/experiments.gemma2.log
python sentence_llm_evaluator.py csabstruct --is-fewshot --num-fewshot-examples 4 --anno-guide-in-user-prompt gemma2 19269 | tee -a _logs/experiments.gemma2.log
python sentence_llm_evaluator.py pubmed200k --is-fewshot --num-fewshot-examples 2 --anno-guide-in-user-prompt gemma2 19269 | tee -a _logs/experiments.gemma2.log
python sentence_llm_evaluator.py pubmed200k --is-fewshot --num-fewshot-examples 4 --anno-guide-in-user-prompt gemma2 19269 | tee -a _logs/experiments.gemma2.log
python sentence_llm_evaluator.py coarsediscourse --is-fewshot --num-fewshot-examples 2 --anno-guide-in-user-prompt gemma2 19269 | tee -a _logs/experiments.gemma2.log
python sentence_llm_evaluator.py coarsediscourse --is-fewshot --num-fewshot-examples 4 --anno-guide-in-user-prompt gemma2 19269 | tee -a _logs/experiments.gemma2.log
python sentence_llm_evaluator.py dailydialog --is-fewshot --num-fewshot-examples 2 --anno-guide-in-user-prompt gemma2 19269 | tee -a _logs/experiments.gemma2.log
python sentence_llm_evaluator.py dailydialog --is-fewshot --num-fewshot-examples 4 --anno-guide-in-user-prompt gemma2 19269 | tee -a _logs/experiments.gemma2.log
python sentence_llm_evaluator.py emotionlines --is-fewshot --num-fewshot-examples 2 --anno-guide-in-user-prompt gemma2 19269 | tee -a _logs/experiments.gemma2.log
python sentence_llm_evaluator.py emotionlines --is-fewshot --num-fewshot-examples 4 --anno-guide-in-user-prompt gemma2 19269 | tee -a _logs/experiments.gemma2.log

python sentence_llm_evaluator.py csabstruct --is-fewshot --num-fewshot-examples 2 --anno-guide-in-user-prompt llama3 19269 | tee -a _logs/experiments.llama3.log
python sentence_llm_evaluator.py csabstruct --is-fewshot --num-fewshot-examples 4 --anno-guide-in-user-prompt llama3 19269 | tee -a _logs/experiments.llama3.log
python sentence_llm_evaluator.py pubmed200k --is-fewshot --num-fewshot-examples 2 --anno-guide-in-user-prompt llama3 19269 | tee -a _logs/experiments.llama3.log
python sentence_llm_evaluator.py pubmed200k --is-fewshot --num-fewshot-examples 4 --anno-guide-in-user-prompt llama3 19269 | tee -a _logs/experiments.llama3.log
python sentence_llm_evaluator.py coarsediscourse --is-fewshot --num-fewshot-examples 2 --anno-guide-in-user-prompt llama3 19269 | tee -a _logs/experiments.llama3.log
python sentence_llm_evaluator.py coarsediscourse --is-fewshot --num-fewshot-examples 4 --anno-guide-in-user-prompt llama3 19269 | tee -a _logs/experiments.llama3.log
python sentence_llm_evaluator.py dailydialog --is-fewshot --num-fewshot-examples 2 --anno-guide-in-user-prompt llama3 19269 | tee -a _logs/experiments.llama3.log
python sentence_llm_evaluator.py dailydialog --is-fewshot --num-fewshot-examples 4 --anno-guide-in-user-prompt llama3 19269 | tee -a _logs/experiments.llama3.log
python sentence_llm_evaluator.py emotionlines --is-fewshot --num-fewshot-examples 2 --anno-guide-in-user-prompt llama3 19269 | tee -a _logs/experiments.llama3.log
python sentence_llm_evaluator.py emotionlines --is-fewshot --num-fewshot-examples 4 --anno-guide-in-user-prompt llama3 19269 | tee -a _logs/experiments.llama3.log

python sentence_llm_evaluator.py csabstruct --is-fewshot --num-fewshot-examples 2 --anno-guide-in-user-prompt mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_evaluator.py csabstruct --is-fewshot --num-fewshot-examples 4 --anno-guide-in-user-prompt mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_evaluator.py pubmed200k --is-fewshot --num-fewshot-examples 2 --anno-guide-in-user-prompt mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_evaluator.py pubmed200k --is-fewshot --num-fewshot-examples 4 --anno-guide-in-user-prompt mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_evaluator.py coarsediscourse --is-fewshot --num-fewshot-examples 2 --anno-guide-in-user-prompt mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_evaluator.py coarsediscourse --is-fewshot --num-fewshot-examples 4 --anno-guide-in-user-prompt mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_evaluator.py dailydialog --is-fewshot --num-fewshot-examples 2 --anno-guide-in-user-prompt mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_evaluator.py dailydialog --is-fewshot --num-fewshot-examples 4 --anno-guide-in-user-prompt mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_evaluator.py emotionlines --is-fewshot --num-fewshot-examples 2 --anno-guide-in-user-prompt mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_evaluator.py emotionlines --is-fewshot --num-fewshot-examples 4 --anno-guide-in-user-prompt mistral 19269 | tee -a _logs/experiments.mistral.log

python sentence_llm_fewshot_evaluator.py csabstruct --num-fewshot-examples 2 mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_fewshot_evaluator.py csabstruct --num-fewshot-examples 4 mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_fewshot_evaluator.py csabstruct --num-fewshot-examples 8 mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_fewshot_evaluator.py csabstruct --num-fewshot-examples 16 mistral 19269 | tee -a _logs/experiments.mistral.log

python sentence_llm_fewshot_evaluator.py pubmed200k --num-fewshot-examples 2 mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_fewshot_evaluator.py pubmed200k --num-fewshot-examples 4 mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_fewshot_evaluator.py pubmed200k --num-fewshot-examples 8 mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_fewshot_evaluator.py pubmed200k --num-fewshot-examples 16 mistral 19269 | tee -a _logs/experiments.mistral.log

python sentence_llm_fewshot_evaluator.py emotionlines --num-fewshot-examples 2 mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_fewshot_evaluator.py emotionlines --num-fewshot-examples 4 mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_fewshot_evaluator.py emotionlines --num-fewshot-examples 8 mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_fewshot_evaluator.py emotionlines --num-fewshot-examples 16 mistral 19269 | tee -a _logs/experiments.mistral.log

python sentence_llm_fewshot_evaluator.py dailydialog --num-fewshot-examples 2 mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_fewshot_evaluator.py dailydialog --num-fewshot-examples 4 mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_fewshot_evaluator.py dailydialog --num-fewshot-examples 8 mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_fewshot_evaluator.py dailydialog --num-fewshot-examples 16 mistral 19269 | tee -a _logs/experiments.mistral.log

python sentence_llm_fewshot_evaluator.py coarsediscourse --num-fewshot-examples 2 mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_fewshot_evaluator.py coarsediscourse --num-fewshot-examples 4 mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_fewshot_evaluator.py coarsediscourse --num-fewshot-examples 8 mistral 19269 | tee -a _logs/experiments.mistral.log
python sentence_llm_fewshot_evaluator.py coarsediscourse --num-fewshot-examples 16 mistral 19269 | tee -a _logs/experiments.mistral.log