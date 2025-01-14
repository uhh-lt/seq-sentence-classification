

### Scientific Paper / Document Understanding / Argumentative Zoning? / Semantic Structuring / Rethorical Analysis

1. CSAbstruct - https://arxiv.org/pdf/1909.04054 - https://github.com/allenai/sequential_sentence_classification - https://huggingface.co/datasets/allenai/csabstruct
   -> Experiments on this DS: https://link.springer.com/article/10.1007/s00799-023-00392-z
   -> Experiments on this DS: https://dl.acm.org/doi/pdf/10.1145/3529372.3530922

2. MuLMS-AZ - https://github.com/boschresearch/mulms-az-codi2023 - https://aclanthology.org/2023.codi-1.1.pdf
   Details: 50 Articles in Domain of Material Science, Labels: Experiment, Results, Exp_Preparation, Exp_Characterization, Background_PriorWork, Explanation, Conclusion, Motivation, Background, Metadata, Caption, Heading, Abstract

3. ART - https://www.aber.ac.uk/en/cs/research/cb/projects/art/art-corpus/ - https://github.com/boschresearch/mulms-az-codi2023/tree/main/data/additional_arg_zoning_datasets

4. AZ-CL - An annotation scheme for discourse-level argumentation in research articles - https://aclanthology.org/E99-1015.pdf see here: https://github.com/boschresearch/mulms-az-codi2023/tree/main/data/additional_arg_zoning_datasets
   Details: Labels: Background, Other, Own, Aim, Textual, Contrast, Basis

5. Dr. Inventor Corpus - A Multi-layered Corpus of Scientific papers - https://aclanthology.org/L16-1492.pdf
   Details: 40 documents in Domain of Computer Graphics, Labels: Challange, Background, Approach, Outcome, Future Work
6. ^-> Investigating the Role of Argumentation in the Rhetorical Analysis of Scientific Publications with Neural Multi-Task Learning Models - https://aclanthology.org/D18-1370.pdf
   Details: Paper von Anne Lauscher, Augmented version von Dr. Inventor Korpus!

7. https://dl.acm.org/doi/pdf/10.1145/3529372.3530922 - Table 1: benchmark datasets
   Pubmed-20k, NICTA-PIBOSO, CSABSTRUCT, CS-Abstracts, Emerald-100k, MAZEA, Dr. Inventor, ART

### Rethorical Roles

1. Semeval 2023 Task 6A: Rethorical Roles Prediction - https://github.com/Legal-NLP-EkStep/rhetorical-role-baseline - https://aclanthology.org/2023.semeval-1.318.pdf - http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.470.pdf - https://huggingface.co/datasets/opennyaiorg/InRhetoricalRoles
   Details: Indian legal documents in english, Label definitions: https://github.com/Legal-NLP-EkStep/rhetorical-role-baseline/blob/main/README.md#appendix
   !! Hier gucken für Model Ideen !!

### Discourse Structure Analysis

### Frame Analysis

1. The Media Frames Corpus: Annotations of Frames Across Issues - https://aclanthology.org/P15-2072.pdf
   -> Experiments on this dataset: Classifying Frames at the Sentence Level in News Articles - https://www.acl-bg.org/proceedings/2017/RANLP%202017/pdf/RANLP070.pdf
2. NO DATASET - Ideological Knowledge Representation: Framing Climate Change in EcoLexicon - https://aclanthology.org/2024.lrec-main.756.pdf
3. NO DATASET - Framing of sustainable agricultural practices by the farming press and its effect on adoption - Sentence Wise frame analysis - https://link.springer.com/article/10.1007/s10460-020-10186-7

### Persuasion Techniques

1. Semeval 2023 Task 3 - https://aclanthology.org/2023.semeval-1.317/ - https://propaganda.math.unipd.it/semeval2023task3/

### Bias Detection

1. Annotating and Analyzing Biased Sentences in News Articles using Crowdsourcing - https://aclanthology.org/2020.lrec-1.184.pdf - https://github.com/skymoonlight/biased-sents-annotation
   Details: Sentence level bias annotations

### Scene Segmentation

1. Shared Task on Scene Segmentation@KONVENS 2021 - http://lsx-events.informatik.uni-wuerzburg.de/stss-2021/
   Details: German Novels
   -> Best Participant - Breaking the Narative - https://ceur-ws.org/Vol-3001/paper6.pdf

### Sentiment Analysis

1. NO DATASET - https://ryanmcd.github.io/papers/hcrf_sentimentECIR2011.pdf

### Opinion Analysis

1. MPQAv2 - https://aclanthology.org/2024.lrec-main.1093.pdf - https://paperswithcode.com/dataset/mpqa-opinion-corpus

### Argument Mining

1. DAS HÖRT SICH NOCH SEHR SEHRR GUT AN https://aclanthology.org/2022.acl-long.162/

### Emotion Analysis

1. EmotionLines - https://paperswithcode.com/dataset/emotionlines - https://doraemon.iis.sinica.edu.tw/emotionlines/index.html
1. DailyDialog/RECCON - https://github.com/declare-lab/RECCON - http://yanran.li/dailydialog - https://paperswithcode.com/dataset/dailydialog

### Topic Segmentation

1. Theory-Driven Analysis of Large Corpora: Semisupervised Topic Classification of the UN Speeches - https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GSDZNV&version=1.0
1. WikiSection - https://aclanthology.org/Q19-1011 - https://github.com/sebastianarnold/WikiSection
   Details: 38K English and German Wikipedia articles from the domains of disease and city, with the topic labeled for each section of the text.
1. Elements - https://groups.csail.mit.edu/rbg/code/mallows/ - https://aclanthology.org/N09-1042.pdf
1. CHOI - https://dl.acm.org/doi/10.5555/974305.974309
1. SpeechTopSeg - probably not useful - https://github.com/sakshishukla1996/SpeechTopSeg - https://arxiv.org/pdf/2409.06222v1
   Works with theses datasets

- https://aclanthology.org/2020.aacl-main.63.pdf
- https://aclanthology.org/2023.emnlp-main.341.pdf
- https://arxiv.org/pdf/2107.09278
- https://arxiv.org/pdf/2209.08626

1. Create an artificial topic dataset by concatenating news from different topics, use random news articles (Similar to CHOI dataset 2000)
1. NO DATASET - Topical Segmentation of Spoken Narratives - https://arxiv.org/pdf/2210.13783
1. DialSeg_77 - Topic-Aware Multi-turn Dialogue Modeling - https://ojs.aaai.org/index.php/AAAI/article/view/17668 - https://github.com/xyease/TADAM/blob/master/examples/DATASET/en/en_dataset_reference_2.txt
1. QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization - https://github.com/Yale-LILY/QMSum - https://aclanthology.org/2021.naacl-main.472/
   -> used in this work: https://peerj.com/articles/cs-1593/

### Text Segmentation

1. Wiki-727k - https://aclanthology.org/N18-2075.pdf - https://github.com/koomri/text-segmentation - https://live.european-language-grid.eu/catalogue/corpus/21630
   Details: documents from English Wikipedia, where the table of contents of each document is used to automatically segment the document
   -> Uses this DS: https://aclanthology.org/2023.emnlp-main.341.pdf - 2023.emnlp-main.249.pdf

### Conversational / Dialog Analysis

https://convokit.cornell.edu/documentation/coarseDiscourse.html

1. Switchboard Dialog Act Corpus
   Details: Speech act tags (https://web.stanford.edu/~jurafsky/ws97/manual.august1.html)
2. Coarse Discourse Sequence Corpus
   Details: majority type: discourse action type by one of the following: question, answer, announcement, agreement, appreciation, disagreement, elaboration, humor
3. SuperDialseg - https://github.com/Coldog2333/SuperDialseg

### Google Datasets

### Political debates

http://www.amber-boydstun.com/uploads/1/0/6/5/106535199/aboydstun2013_playing_to_the_crowd_agenda_control_in_presidential_debates.pdf
-> uses this DS: https://aclanthology.org/P12-1009.pdf
`