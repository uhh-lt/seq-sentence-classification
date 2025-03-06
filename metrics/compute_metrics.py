from typing import Dict, List
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
import pandas as pd


def to_bio_format(labels: List[List[str]]) -> List[List[str]]:
    bio_labels = []
    for label_list in labels:
        bio_label_list = []
        prev_label = "o"
        for label in label_list:
            if label == "o":
                bio_label_list.append("O")
            elif label != prev_label:
                bio_label_list.append("B-" + label)
            else:
                bio_label_list.append("I-" + label)
            prev_label = label
        bio_labels.append(bio_label_list)
    return bio_labels


def compute_metrics(
    id2tag: Dict[int, str], preds: List[List[int]], golds: List[List[int]]
):
    # convert preds and golds to BIO format
    preds_bio = to_bio_format([[id2tag[p] for p in preds] for preds in preds])
    golds_bio = to_bio_format([[id2tag[t] for t in tags] for tags in golds])

    # compute metrics
    acc = accuracy_score(golds_bio, preds_bio)
    report = classification_report(golds_bio, preds_bio, output_dict=True)
    df_report = pd.DataFrame(report)
    precision, recall, f1, support = [
        round(x * 100.0, 2) for x in df_report["weighted avg"].tolist()
    ]

    return precision, recall, f1, acc
