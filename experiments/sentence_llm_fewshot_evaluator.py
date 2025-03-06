from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import typer
import rootutils
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from pydantic import BaseModel, Field
from openai.types.chat import ChatCompletionMessageParam
import json

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from experiments.llm_evaluator import ModelsEnum, LLMEvaluator  # noqa: E402

app = typer.Typer()

NUM_EXAMPLES = 500
DEBUG_PROMPT = False


class TextAnnotation(BaseModel):
    text_id: int = Field(description="The id of the text")
    reason: Optional[str] = Field(None, description="The reason for the classification")
    category: str = Field(
        description="The classification of the text according to the annotation guidelines"
    )


class SentenceLLMFewshotEvaluator(LLMEvaluator):
    def __init__(
        self,
        project_name: str,
        project_details: str,
        system_prompt_template: str,
        user_prompt_template: str,
        documents: List[List[str]],
        labels: List[List[str]],
        label_dict: dict,
        fewshot_documents: Dict[str, List[str]],
        model: ModelsEnum,
        port: int,
        lang: str,
        dataset_name: str,
        task_name: str,
        output_dir_path: Path,
        report_path: Path,
        unwanted_labels: List[str] = [],
    ):
        # call parent
        super(SentenceLLMFewshotEvaluator, self).__init__(
            model=model,
            port=port,
            lang=lang,
            dataset_name=dataset_name,
            task_name=task_name,
            output_dir_path=output_dir_path,
            report_path=report_path,
        )

        # assert that documents and labels are correct
        assert len(documents) == len(
            labels
        ), "The number of documents and labels must be the same."
        unique_labels = []
        for document, label in zip(documents, labels):
            assert len(document) == len(
                label
            ), "The number of sentences in each document and label must be the same."
            for sentence in document:
                assert (
                    sentence.count("\n") == 0
                ), "The sentence must not contain newlines."
            unique_labels.extend(label)
        unique_labels = set(unique_labels)
        unique_labels = unique_labels.difference(set(unwanted_labels))

        # convert all labels to lowercase
        self.labels = [[label.lower() for label in lls] for lls in labels]
        self.documents = documents

        # fewshot_documents is a dict of label to list of examples
        # conver all labels to lowercase
        self.fewshot_documents = {
            label.lower(): examples for label, examples in fewshot_documents.items()
        }

        # assert that there are examples for each label
        for unique_label in unique_labels:
            assert (
                unique_label in self.fewshot_documents
            ), f"The label '{unique_label}' must have fewshot examples."

        # assert that there are the same number of examples for each label
        num_examples = len(list(self.fewshot_documents.values())[0])
        for label, examples in self.fewshot_documents.items():
            assert (
                len(examples) == num_examples
            ), f"The number of examples differs for the label {label}."

        # convert all label_dict keys to lowercase
        self.label_dict = {key.lower(): value for key, value in label_dict.items()}
        # assert that the label_dict is correct
        for unique_label in unique_labels:
            assert (
                unique_label in self.label_dict
            ), f"The label '{unique_label}' must be in the label_dict."

        # build annotation guidelines with the help of the label_dict, fewshot_labels and fewshot_documents
        self.annotation_guidelines = ""
        for label, description in label_dict.items():
            examples = self.fewshot_documents[label]

            self.annotation_guidelines += f"{label.lower()}: {description.strip()}\n For example:\n"
            assert (
                description.count("\n") == 0
            ), "The description must not contain newlines."
            for example in examples:
                self.annotation_guidelines += f"  - {example.strip()}\n"
                assert (
                    example.count("\n") == 0
                ), "The example must not contain newlines."
            self.annotation_guidelines += "\n"
        self.annotation_guidelines = self.annotation_guidelines.strip()

        # assert that prompt templates have the correct placeholders
        system_placeholders = [
            "project_name",
            "project_details",
        ]
        for placeholder in system_placeholders:
            assert (
                "<" + placeholder + ">" in system_prompt_template
            ), f"The system_prompt_template must contain the <{placeholder}> placeholder."
        user_placeholders = [
            "annotation_guidelines",
        ]
        for placeholder in user_placeholders:
            assert (
                "<" + placeholder + ">" in user_prompt_template
            ), f"The user_prompt_template must contain the <{placeholder}> placeholder."
        assert (
            "{document}" in user_prompt_template
        ), "The user_prompt_template must contain the {document} placeholder."

        # build system prompt
        assert len(project_name) > 0, "The project_name must not be empty."
        assert (
            project_name.count("\n") == 0
        ), "The project_name must not contain newlines."

        assert len(project_details) > 0, "The project_details must not be empty."
        assert (
            project_details.count("\n") == 0
        ), "The project_details must not contain newlines."
        self.system_prompt = (
            system_prompt_template.replace("<project_name>", project_name)
            .replace("<project_details>", project_details)
            .strip()
        )

        # build user prompt template
        self.user_prompt_template = user_prompt_template.replace(
                "<annotation_guidelines>", self.annotation_guidelines
            ).strip()

        print("---- Using this label dict ----")
        print(self.label_dict)
        print()
        print("---- Using this annotation guidelines ----")
        print(self.annotation_guidelines)
        print()
        print("---- Using this system prompt ----")
        print(self.system_prompt)
        print()
        print("---- Using this user prompt template ----")
        print(self.user_prompt_template)
        print()

        self.unwanted_labels = unwanted_labels

    def _parse_classification(self, classification: str) -> str:
        result = classification.lower()
        if result not in self.label_dict.keys():
            result = "o"
        return result

    def _parse_json_response(self, response: List[TextAnnotation]) -> Dict[int, str]:
        parsed_result = {}
        for resp in response:
            parsed_result[resp.text_id] = self._parse_classification(resp.category)

        return parsed_result

    def _build_document_string(self, document: List[str]) -> str:
        return "\n".join([f"{i+1}: {sentence}" for i, sentence in enumerate(document)])

    def _prompt_ollama(self, document: str):
        try:
            messages: List[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": self.user_prompt_template.format(
                        document=document
                    ).strip(),
                },
            ]

            if DEBUG_PROMPT:
                for message in messages:
                    print(f"---- {message['role']} ----")
                    print(message["content"])  # type: ignore
                    print()
                print("---- END ----")
                exit()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_model=List[TextAnnotation],
            )
            response_str = "\n".join([resp.model_dump_json() for resp in response])
            return response_str, self._parse_json_response(response)
        except Exception as e:  # noqa: F841
            return "", {}

    def _evaluate(self):
        predictions = []
        messages = []
        for document, label in tqdm(
            zip(self.documents, self.labels), desc="Evaluating"
        ):
            document_string = self._build_document_string(document)
            message, sent_classification = self._prompt_ollama(document_string)

            # convert sentence_classification to list
            sent_classification = [
                sent_classification.get(i + 1, "o") for i in range(len(document))
            ]

            predictions.append(sent_classification)
            messages.append(message)

        # store the evaluation results in a csv file
        pd.DataFrame(
            {
                "document": self.documents,
                "Label": self.labels,
                "Prediction": predictions,
                "Message": messages,
            }
        ).to_parquet(self.output_file_path, index=False)

    def _report(self):
        # read the evaluation results
        df = pd.read_parquet(self.output_file_path)
        results_len = len(df)

        assert df.size > 0, "The evaluation results are empty."

        # remove columns where the prediction is None or an empty list
        df = df[df["Prediction"].notna()]
        df = df[df["Prediction"].apply(lambda x: len(x) > 0)]
        results_filtered_len = len(df)

        # count None values
        none_count = results_len - results_filtered_len
        print(f"Total count: {results_len}")
        print(f"Filtered count: {results_filtered_len}")
        print(
            f"None count: {none_count}, None percentage: {(none_count / results_len) * 100:.2f}%"
        )

        # lowercase preds and labels
        df["Label"] = df["Label"].apply(lambda x: [pred.lower() for pred in x])
        df["Prediction"] = df["Prediction"].apply(
            lambda x: [pred.lower() for pred in x]
        )

        # convert the golds and preds to the bio format
        golds = [list(x) for x in df["Label"].to_list()]
        preds = [list(x) for x in df["Prediction"].to_list()]

        def filter_unwanted_labels(golds, preds, unwanted_labels):
            if len(unwanted_labels) == 0:
                return golds, preds

            golds_filtered = []
            preds_filtered = []

            for gold, pred in zip(golds, preds):
                gold_filtered = []
                pred_filtered = []
                for g, p in zip(gold, pred):
                    if g not in unwanted_labels:
                        gold_filtered.append(g)
                        pred_filtered.append(p)
                golds_filtered.append(gold_filtered)
                preds_filtered.append(pred_filtered)

            return golds_filtered, preds_filtered

        def to_bio_format(labels):
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

        golds_filtered, preds_filtered = filter_unwanted_labels(
            golds, preds, self.unwanted_labels
        )

        golds_bio = to_bio_format(golds_filtered)
        preds_bio = to_bio_format(preds_filtered)

        # compute metrics
        f1 = f1_score(golds_bio, preds_bio)
        acc = accuracy_score(golds_bio, preds_bio)
        report = classification_report(golds_bio, preds_bio, output_dict=True)
        df_report = pd.DataFrame(report)

        # Print report
        print("F1-Score: ", f1)
        print("Accuracy: ", acc)
        print(classification_report(golds_bio, preds_bio))

        # extract most important information
        precision, recall, f1, support = [
            round(x * 100.0, 2) for x in df_report["weighted avg"].tolist()
        ]

        # write reports
        df_report.transpose().to_csv(self.output_file_path.with_suffix(".report.csv"))
        self._add_results_to_report(
            {
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Accuracy": round(acc * 100.0, 2),
            }
        )


@app.command()
def csabstruct(
    model: ModelsEnum,
    port: int,
    report_only: bool = False,
    num_fewshot_examples: int = 2,
):
    from experiments.csabstruct.details import (
        label_dict,
        system_prompt_template,
        user_prompt_with_guidelines_template,
        project_name,
        project_details,
    )

    # load dataset
    df = pd.read_parquet("datasets/csabstruct/test.parquet")
    documents: List[List[str]] = [
        list(document_sentences) for document_sentences in df["sentences"].tolist()
    ][:NUM_EXAMPLES]
    labels: List[List[str]] = [list(labels) for labels in df["labels"].tolist()][
        :NUM_EXAMPLES
    ]

    # load fewshot examples
    with open(f"datasets/csabstruct/few_shot_nAll_k{num_fewshot_examples}.json", "r") as json_file:
        fewshot_docs = json.load(json_file)

    # start evaluator
    evaluator = SentenceLLMFewshotEvaluator(
        model=model,
        port=port,
        project_name=project_name,
        project_details=project_details,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_with_guidelines_template,
        fewshot_documents=fewshot_docs,
        lang="en",
        task_name=f"few-shot-{num_fewshot_examples}",
        dataset_name="csabstruct",
        output_dir_path=Path("experiments/csabstruct"),
        report_path=Path("experiments/csabstruct/report.csv"),
        documents=documents,
        labels=labels,
        label_dict=label_dict,
    )
    evaluator.start(report_only=report_only)


@app.command()
def pubmed200k(
    model: ModelsEnum,
    port: int,
    report_only: bool = False,
    num_fewshot_examples: int = 2,
):
    from experiments.pubmed200k.details import (
        label_dict,
        system_prompt_template,
        user_prompt_with_guidelines_template,
        project_name,
        project_details,
    )

    # load dataset
    df = pd.read_parquet("datasets/pubmed200k/test.parquet")
    documents: List[List[str]] = [
        list(document_sentences) for document_sentences in df["sentences"].tolist()
    ][:NUM_EXAMPLES]
    labels: List[List[str]] = [list(labels) for labels in df["labels"].tolist()][
        :NUM_EXAMPLES
    ]

    # load fewshot examples
    with open(f"datasets/pubmed200k/few_shot_nAll_k{num_fewshot_examples}.json", "r") as json_file:
        fewshot_docs = json.load(json_file)

    # start evaluator
    evaluator = SentenceLLMFewshotEvaluator(
        model=model,
        port=port,
        project_name=project_name,
        project_details=project_details,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_with_guidelines_template,
        fewshot_documents=fewshot_docs,
        lang="en",
        task_name=f"few-shot-{num_fewshot_examples}",
        dataset_name="pubmed200k",
        output_dir_path=Path("experiments/pubmed200k"),
        report_path=Path("experiments/pubmed200k/report.csv"),
        documents=documents,
        labels=labels,
        label_dict=label_dict,
    )
    evaluator.start(report_only=report_only)


@app.command()
def coarsediscourse(
    model: ModelsEnum,
    port: int,
    report_only: bool = False,
    num_fewshot_examples: int = 2,
):
    from experiments.coarsediscourse.details import (
        label_dict,
        system_prompt_template,
        user_prompt_with_guidelines_template,
        project_name,
        project_details,
    )

    # load dataset
    df = pd.read_parquet("datasets/coarsediscourse/coursediscourse_test.parquet")
    documents: List[List[str]] = [
        list(document_sentences) for document_sentences in df["sentences"].tolist()
    ][:NUM_EXAMPLES]
    labels: List[List[str]] = [list(labels) for labels in df["labels"].tolist()][
        :NUM_EXAMPLES
    ]

    # load fewshot examples
    with open(f"datasets/coarsediscourse/few_shot_nAll_k{num_fewshot_examples}.json", "r") as json_file:
        fewshot_docs = json.load(json_file)

    # start evaluator
    evaluator = SentenceLLMFewshotEvaluator(
        model=model,
        port=port,
        project_name=project_name,
        project_details=project_details,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_with_guidelines_template,
        fewshot_documents=fewshot_docs,
        lang="en",
        task_name=f"few-shot-{num_fewshot_examples}",
        dataset_name="coarsediscourse",
        output_dir_path=Path("experiments/coarsediscourse"),
        report_path=Path("experiments/coarsediscourse/report.csv"),
        documents=documents,
        labels=labels,
        label_dict=label_dict,
    )
    evaluator.start(report_only=report_only)


@app.command()
def dailydialog(
    model: ModelsEnum,
    port: int,
    report_only: bool = False,
    num_fewshot_examples: int = 2,
):
    from experiments.daily_dialog.details import (
        label_dict,
        system_prompt_template,
        user_prompt_with_guidelines_template,
        project_name,
        project_details,
    )

    # load dataset
    df = pd.read_parquet("datasets/daily_dialog/dailydialog_test.parquet")
    documents: List[List[str]] = [
        list(document_sentences) for document_sentences in df["sentences"].tolist()
    ][:NUM_EXAMPLES]
    labels: List[List[str]] = [list(labels) for labels in df["labels"].tolist()][
        :NUM_EXAMPLES
    ]

    # load fewshot examples
    with open(f"datasets/daily_dialog/few_shot_nAll_k{num_fewshot_examples}.json", "r") as json_file:
        fewshot_docs = json.load(json_file)

    # start evaluator
    evaluator = SentenceLLMFewshotEvaluator(
        model=model,
        port=port,
        project_name=project_name,
        project_details=project_details,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_with_guidelines_template,
        fewshot_documents=fewshot_docs,
        lang="en",
        task_name=f"few-shot-{num_fewshot_examples}",
        dataset_name="daily_dialog",
        output_dir_path=Path("experiments/daily_dialog"),
        report_path=Path("experiments/daily_dialog/report.csv"),
        documents=documents,
        labels=labels,
        label_dict=label_dict,
    )
    evaluator.start(report_only=report_only)


@app.command()
def emotionlines(
    model: ModelsEnum,
    port: int,
    report_only: bool = False,
    num_fewshot_examples: int = 2,
):
    from experiments.emotion_lines.details import (
        label_dict,
        system_prompt_template,
        user_prompt_with_guidelines_template,
        project_name,
        project_details,
    )

    # load dataset
    df = pd.read_parquet("datasets/emotion_lines/friends_test.parquet")
    documents: List[List[str]] = [
        list(document_sentences) for document_sentences in df["sentences"].tolist()
    ][:NUM_EXAMPLES]
    labels: List[List[str]] = [list(labels) for labels in df["labels"].tolist()][
        :NUM_EXAMPLES
    ]

    # load fewshot examples
    with open(f"datasets/emotion_lines/few_shot_nAll_k{num_fewshot_examples}.json", "r") as json_file:
        fewshot_docs = json.load(json_file)

    # start evaluator
    evaluator = SentenceLLMFewshotEvaluator(
        model=model,
        port=port,
        project_name=project_name,
        project_details=project_details,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_with_guidelines_template,
        fewshot_documents=fewshot_docs,
        lang="en",
        task_name=f"few-shot-{num_fewshot_examples}",
        dataset_name="emotion_lines",
        output_dir_path=Path("experiments/emotion_lines"),
        report_path=Path("experiments/emotion_lines/report.csv"),
        documents=documents,
        labels=labels,
        label_dict=label_dict,
        unwanted_labels=["non-neutral"],
    )
    evaluator.start(report_only=report_only)


if __name__ == "__main__":
    app()
