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


class SentenceLLMEvaluator(LLMEvaluator):
    def __init__(
        self,
        project_name: str,
        project_details: str,
        system_prompt_template: str,
        system_prompt_with_guidelines_template: str,
        user_prompt_template: str,
        user_prompt_with_guidelines_template: str,
        is_anno_guide_in_user_prompt: bool,
        documents: List[List[str]],
        labels: List[List[str]],
        label_dict: dict,
        is_fewshot: bool,
        is_fewshot_json: bool,
        fewshot_documents: List[List[str]],
        fewshot_labels: List[List[str]],
        fewshot_reasons: List[List[str]],
        use_fewshot_reasons: bool,
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
        super(SentenceLLMEvaluator, self).__init__(
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

        if is_fewshot:
            # assert that fewshot documents and labels are correct
            assert len(fewshot_documents) == len(
                fewshot_labels
            ), "The number of fewshot_documents and fewshot_labels must be the same."
            for document, label in zip(fewshot_documents, fewshot_labels):
                assert (
                    len(document) == len(label)
                ), "The number of sentences in each document and label must be the same."
                for sentence in document:
                    assert (
                        sentence.count("\n") == 0
                    ), "The sentence must not contain newlines."

        self.is_fewshot = is_fewshot
        self.is_fewshot_json = is_fewshot_json
        self.fewshot_labels = [
            [label.lower() for label in lls] for lls in fewshot_labels
        ]
        self.fewshot_documents = fewshot_documents
        self.fewshot_reasons = fewshot_reasons
        self.use_fewshot_reasons = use_fewshot_reasons

        # convert all label_dict keys to lowercase
        self.label_dict = {key.lower(): value for key, value in label_dict.items()}
        # assert that the label_dict is correct
        for unique_label in unique_labels:
            assert (
                unique_label in self.label_dict
            ), f"The label '{unique_label}' must be in the label_dict."

        # build annotation guidelines with the help of the label_dict
        self.annotation_guidelines = ""
        for label, description in label_dict.items():
            self.annotation_guidelines += f"{label.lower()}\n"
            assert (
                description.count("\n") == 0
            ), "The description must not contain newlines."
            self.annotation_guidelines += f"{description}\n\n"
        self.annotation_guidelines = self.annotation_guidelines.strip()

        # assert that prompt templates has the correct placeholders
        system_placeholders = [
            "project_name",
            "project_details",
        ]
        for placeholder in system_placeholders:
            assert (
                "<" + placeholder + ">" in system_prompt_template
            ), f"The system_prompt_template must contain the <{placeholder}> placeholder."
            assert (
                "<" + placeholder + ">" in system_prompt_with_guidelines_template
            ), f"The system_prompt_with_guidelines_template must contain the <{placeholder}> placeholder."

        assert (
            "{document}" in user_prompt_template
        ), "The user_prompt_template must contain the {document} placeholder."
        assert (
            "{document}" in user_prompt_with_guidelines_template
        ), "The user_prompt_with_guidelines_template must contain the {document} placeholder."

        assert (
            "<annotation_guidelines>" in system_prompt_with_guidelines_template
        ), 'The system_prompt_with_guidelines_template must contain the "<annotation_guidelines>" placeholder.'
        assert (
            "<annotation_guidelines>" in user_prompt_with_guidelines_template
        ), 'The user_prompt_with_guidelines_template must contain the "<annotation_guidelines>" placeholder.'

        # build system prompt
        assert len(project_name) > 0, "The project_name must not be empty."
        assert (
            project_name.count("\n") == 0
        ), "The project_name must not contain newlines."

        assert len(project_details) > 0, "The project_details must not be empty."
        assert (
            project_details.count("\n") == 0
        ), "The project_details must not contain newlines."

        self.is_anno_guidelines_in_user_prompt = is_anno_guide_in_user_prompt

        if self.is_anno_guidelines_in_user_prompt:
            self.system_prompt = (
                system_prompt_template.replace("<project_name>", project_name)
                .replace("<project_details>", project_details)
                .strip()
            )
        else:
            self.system_prompt = (
                system_prompt_with_guidelines_template.replace(
                    "<project_name>", project_name
                )
                .replace("<project_details>", project_details)
                .replace("<annotation_guidelines>", self.annotation_guidelines)
                .strip()
            )

        self.user_prompt_template = user_prompt_template
        self.user_prompt_with_guidelines_template = (
            user_prompt_with_guidelines_template.replace(
                "<annotation_guidelines>", self.annotation_guidelines
            ).strip()
        )

        print("---- Using this system prompt ----")
        print(self.system_prompt)
        print()
        print("---- Using this user prompt template ----")
        print(self.user_prompt_template)
        print()
        print("---- Using this user prompt template with guidelines ----")
        print(self.user_prompt_with_guidelines_template)
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

    def _build_answer_string(
        self, labels: List[str], reasons: Optional[List[str]] = None
    ) -> str:
        if self.use_fewshot_reasons and reasons is None:
            raise ValueError(
                "Reasons must be provided when use_fewshot_reasons is True."
            )
            exit()
        elif self.use_fewshot_reasons and reasons is not None:
            return "\n".join(
                [
                    f"{idx+1}: reason={reason} category={label}"
                    for idx, (label, reason) in enumerate(zip(labels, reasons))
                ]
            )
        else:
            return "\n".join([f"{idx+1}: {label}" for idx, label in enumerate(labels)])

    def _build_json_answer_string(
        self, labels: List[str], reasons: Optional[List[str]] = None
    ) -> str:
        if self.use_fewshot_reasons and reasons is None:
            raise ValueError(
                "Reasons must be provided when use_fewshot_reasons is True."
            )
            exit()
        elif self.use_fewshot_reasons and reasons is not None:
            annotations = [
                TextAnnotation(
                    text_id=idx + 1,
                    reason=reason,
                    category=label,
                )
                for idx, (label, reason) in enumerate(zip(labels, reasons))
            ]
        else:
            annotations = [
                TextAnnotation(
                    text_id=idx + 1,
                    reason=None,
                    category=label,
                )
                for idx, label in enumerate(labels)
            ]

        message = "[\n"
        for annotation in annotations:
            message += (
                annotation.model_dump_json(
                    exclude=None if self.use_fewshot_reasons else "reason"  # type: ignore
                )
                + ",\n"
            )
        message = message[:-2] + "\n]"
        return message

    def _construct_fewshot_messages(self):
        if not self.is_fewshot:
            return []

        # remove last line from user prompt template
        user_prompt_template = "\n".join(
            self.user_prompt_template.strip().split("\n")[:-1]
        )

        messages = []

        if self.use_fewshot_reasons:
            assert len(self.fewshot_reasons) == len(
                self.fewshot_documents
            ), "The number of fewshot reasons and fewshot documents must be the same."

            for sentences, labels, reasons in zip(
                self.fewshot_documents, self.fewshot_labels, self.fewshot_reasons
            ):
                document_string = self._build_document_string(sentences)
                messages.append(
                    {
                        "role": "user",
                        "content": user_prompt_template.format(
                            document=document_string
                        ).strip(),
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": self._build_json_answer_string(labels, reasons)
                        if self.is_fewshot_json
                        else self._build_answer_string(labels, reasons),
                    }
                )
        else:
            for sentences, labels in zip(self.fewshot_documents, self.fewshot_labels):
                document_string = self._build_document_string(sentences)
                messages.append(
                    {
                        "role": "user",
                        "content": user_prompt_template.format(
                            document=document_string
                        ).strip(),
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": self._build_json_answer_string(labels)
                        if self.is_fewshot_json
                        else self._build_answer_string(labels),
                    }
                )

        return messages

    def _prompt_ollama(self, document: str):
        # select correct user prompt template
        user_prompt_template = (
            self.user_prompt_with_guidelines_template
            if self.is_anno_guidelines_in_user_prompt
            else self.user_prompt_template
        )

        try:
            if self.is_fewshot:
                messages: List[ChatCompletionMessageParam] = [
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    }
                ]
                messages.extend(self._construct_fewshot_messages())
                messages.append(
                    {
                        "role": "user",
                        "content": user_prompt_template.format(
                            document=document
                        ).strip(),
                    }
                )
            else:
                messages: List[ChatCompletionMessageParam] = [
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt_template.format(
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
    is_fewshot: bool = False,
    num_fewshot_examples: int = 2,
    anno_guide_in_user_prompt: bool = False,
    use_fewshot_reasons: bool = False,
    is_fewshot_json: bool = False,
):
    from experiments.csabstruct.details import (
        label_dict,
        system_prompt_template,
        system_prompt_with_guidelines_template,
        user_prompt_template,
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

    if is_fewshot:
        fewshot_df = pd.read_parquet(
            f"datasets/csabstruct/few_shot_nAll_k{num_fewshot_examples}.parquet"
        )
        fewshot_docs: List[List[str]] = [
            list(document_sentences)
            for document_sentences in fewshot_df["sentences"].tolist()
        ]
        fewshot_labels: List[List[str]] = [
            list(labels) for labels in fewshot_df["labels"].tolist()
        ]
        if use_fewshot_reasons:
            fewshot_reasons: List[List[str]] = [
                list(labels) for labels in fewshot_df[f"reasons_{model}"].tolist()
            ]
        else:
            fewshot_reasons = []

    task_name = "few-shot" if is_fewshot else "zero-shot"
    if is_fewshot:
        task_name += f"-{num_fewshot_examples}"
    if use_fewshot_reasons:
        task_name += "-with-reasons"
    if anno_guide_in_user_prompt:
        task_name += "-with-anno-guide-in-user-prompt"
    if is_fewshot_json:
        task_name += "-is-json"

    # start evaluator
    evaluator = SentenceLLMEvaluator(
        model=model,
        port=port,
        project_name=project_name,
        project_details=project_details,
        system_prompt_template=system_prompt_template,
        system_prompt_with_guidelines_template=system_prompt_with_guidelines_template,
        user_prompt_template=user_prompt_template,
        user_prompt_with_guidelines_template=user_prompt_with_guidelines_template,
        is_anno_guide_in_user_prompt=anno_guide_in_user_prompt,
        is_fewshot=is_fewshot,
        is_fewshot_json=is_fewshot_json,
        fewshot_documents=fewshot_docs if is_fewshot else [],
        fewshot_labels=fewshot_labels if is_fewshot else [],
        fewshot_reasons=fewshot_reasons if is_fewshot and use_fewshot_reasons else [],
        use_fewshot_reasons=use_fewshot_reasons,
        lang="en",
        task_name=task_name,
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
    is_fewshot: bool = False,
    num_fewshot_examples: int = 2,
    anno_guide_in_user_prompt: bool = False,
    use_fewshot_reasons: bool = False,
    is_fewshot_json: bool = False,
):
    from experiments.pubmed200k.details import (
        label_dict,
        system_prompt_template,
        system_prompt_with_guidelines_template,
        user_prompt_template,
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

    if is_fewshot:
        fewshot_df = pd.read_parquet(
            f"datasets/pubmed200k/few_shot_nAll_k{num_fewshot_examples}.parquet"
        )
        fewshot_docs: List[List[str]] = [
            list(document_sentences)
            for document_sentences in fewshot_df["sentences"].tolist()
        ]
        fewshot_labels: List[List[str]] = [
            list(labels) for labels in fewshot_df["labels"].tolist()
        ]
        if use_fewshot_reasons:
            fewshot_reasons: List[List[str]] = [
                list(labels) for labels in fewshot_df[f"reasons_{model}"].tolist()
            ]
        else:
            fewshot_reasons = []

    task_name = "few-shot" if is_fewshot else "zero-shot"
    if is_fewshot:
        task_name += f"-{num_fewshot_examples}"
    if use_fewshot_reasons:
        task_name += "-with-reasons"
    if anno_guide_in_user_prompt:
        task_name += "-with-anno-guide-in-user-prompt"
    if is_fewshot_json:
        task_name += "-is-json"

    # start evaluator
    evaluator = SentenceLLMEvaluator(
        model=model,
        port=port,
        project_name=project_name,
        project_details=project_details,
        system_prompt_template=system_prompt_template,
        system_prompt_with_guidelines_template=system_prompt_with_guidelines_template,
        user_prompt_template=user_prompt_template,
        user_prompt_with_guidelines_template=user_prompt_with_guidelines_template,
        is_anno_guide_in_user_prompt=anno_guide_in_user_prompt,
        is_fewshot=is_fewshot,
        is_fewshot_json=is_fewshot_json,
        fewshot_documents=fewshot_docs if is_fewshot else [],
        fewshot_labels=fewshot_labels if is_fewshot else [],
        fewshot_reasons=fewshot_reasons if is_fewshot and use_fewshot_reasons else [],
        use_fewshot_reasons=use_fewshot_reasons,
        lang="en",
        task_name=task_name,
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
    is_fewshot: bool = False,
    num_fewshot_examples: int = 2,
    anno_guide_in_user_prompt: bool = False,
    use_fewshot_reasons: bool = False,
    is_fewshot_json: bool = False,
):
    from experiments.coarsediscourse.details import (
        label_dict,
        system_prompt_template,
        system_prompt_with_guidelines_template,
        user_prompt_template,
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

    if is_fewshot:
        fewshot_df = pd.read_parquet(
            f"datasets/coarsediscourse/few_shot_nAll_k{num_fewshot_examples}.parquet"
        )
        fewshot_docs: List[List[str]] = [
            list(document_sentences)
            for document_sentences in fewshot_df["sentences"].tolist()
        ]
        fewshot_labels: List[List[str]] = [
            list(labels) for labels in fewshot_df["labels"].tolist()
        ]
        if use_fewshot_reasons:
            fewshot_reasons: List[List[str]] = [
                list(labels) for labels in fewshot_df[f"reasons_{model}"].tolist()
            ]
        else:
            fewshot_reasons = []

    task_name = "few-shot" if is_fewshot else "zero-shot"
    if is_fewshot:
        task_name += f"-{num_fewshot_examples}"
    if use_fewshot_reasons:
        task_name += "-with-reasons"
    if anno_guide_in_user_prompt:
        task_name += "-with-anno-guide-in-user-prompt"
    if is_fewshot_json:
        task_name += "-is-json"

    # start evaluator
    evaluator = SentenceLLMEvaluator(
        model=model,
        port=port,
        project_name=project_name,
        project_details=project_details,
        system_prompt_template=system_prompt_template,
        system_prompt_with_guidelines_template=system_prompt_with_guidelines_template,
        user_prompt_template=user_prompt_template,
        user_prompt_with_guidelines_template=user_prompt_with_guidelines_template,
        is_anno_guide_in_user_prompt=anno_guide_in_user_prompt,
        is_fewshot=is_fewshot,
        is_fewshot_json=is_fewshot_json,
        fewshot_documents=fewshot_docs if is_fewshot else [],
        fewshot_labels=fewshot_labels if is_fewshot else [],
        fewshot_reasons=fewshot_reasons if is_fewshot and use_fewshot_reasons else [],
        use_fewshot_reasons=use_fewshot_reasons,
        lang="en",
        task_name=task_name,
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
    is_fewshot: bool = False,
    num_fewshot_examples: int = 2,
    anno_guide_in_user_prompt: bool = False,
    use_fewshot_reasons: bool = False,
    is_fewshot_json: bool = False,
):
    from experiments.daily_dialog.details import (
        label_dict,
        system_prompt_template,
        system_prompt_with_guidelines_template,
        user_prompt_template,
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

    if is_fewshot:
        fewshot_df = pd.read_parquet(
            f"datasets/daily_dialog/few_shot_nAll_k{num_fewshot_examples}.parquet"
        )
        fewshot_docs: List[List[str]] = [
            list(document_sentences)
            for document_sentences in fewshot_df["sentences"].tolist()
        ]
        fewshot_labels: List[List[str]] = [
            list(labels) for labels in fewshot_df["labels"].tolist()
        ]
        if use_fewshot_reasons:
            fewshot_reasons: List[List[str]] = [
                list(labels) for labels in fewshot_df[f"reasons_{model}"].tolist()
            ]
        else:
            fewshot_reasons = []

    task_name = "few-shot" if is_fewshot else "zero-shot"
    if is_fewshot:
        task_name += f"-{num_fewshot_examples}"
    if use_fewshot_reasons:
        task_name += "-with-reasons"
    if anno_guide_in_user_prompt:
        task_name += "-with-anno-guide-in-user-prompt"
    if is_fewshot_json:
        task_name += "-is-json"

    # start evaluator
    evaluator = SentenceLLMEvaluator(
        model=model,
        port=port,
        project_name=project_name,
        project_details=project_details,
        system_prompt_template=system_prompt_template,
        system_prompt_with_guidelines_template=system_prompt_with_guidelines_template,
        user_prompt_template=user_prompt_template,
        user_prompt_with_guidelines_template=user_prompt_with_guidelines_template,
        is_anno_guide_in_user_prompt=anno_guide_in_user_prompt,
        is_fewshot=is_fewshot,
        is_fewshot_json=is_fewshot_json,
        fewshot_documents=fewshot_docs if is_fewshot else [],
        fewshot_labels=fewshot_labels if is_fewshot else [],
        fewshot_reasons=fewshot_reasons if is_fewshot and use_fewshot_reasons else [],
        use_fewshot_reasons=use_fewshot_reasons,
        lang="en",
        task_name=task_name,
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
    is_fewshot: bool = False,
    num_fewshot_examples: int = 2,
    anno_guide_in_user_prompt: bool = False,
    use_fewshot_reasons: bool = False,
    is_fewshot_json: bool = False,
):
    from experiments.emotion_lines.details import (
        label_dict,
        system_prompt_template,
        system_prompt_with_guidelines_template,
        user_prompt_template,
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

    if is_fewshot:
        fewshot_df = pd.read_parquet(
            f"datasets/emotion_lines/few_shot_nAll_k{num_fewshot_examples}.parquet"
        )
        fewshot_docs: List[List[str]] = [
            list(document_sentences)
            for document_sentences in fewshot_df["sentences"].tolist()
        ]
        fewshot_labels: List[List[str]] = [
            list(labels) for labels in fewshot_df["labels"].tolist()
        ]
        if use_fewshot_reasons:
            fewshot_reasons: List[List[str]] = [
                list(labels) for labels in fewshot_df[f"reasons_{model}"].tolist()
            ]
        else:
            fewshot_reasons = []

    task_name = "few-shot" if is_fewshot else "zero-shot"
    if is_fewshot:
        task_name += f"-{num_fewshot_examples}"
    if use_fewshot_reasons:
        task_name += "-with-reasons"
    if anno_guide_in_user_prompt:
        task_name += "-with-anno-guide-in-user-prompt"
    if is_fewshot_json:
        task_name += "-is-json"

    # start evaluator
    evaluator = SentenceLLMEvaluator(
        model=model,
        port=port,
        project_name=project_name,
        project_details=project_details,
        system_prompt_template=system_prompt_template,
        system_prompt_with_guidelines_template=system_prompt_with_guidelines_template,
        user_prompt_template=user_prompt_template,
        user_prompt_with_guidelines_template=user_prompt_with_guidelines_template,
        is_anno_guide_in_user_prompt=anno_guide_in_user_prompt,
        is_fewshot=is_fewshot,
        is_fewshot_json=is_fewshot_json,
        fewshot_documents=fewshot_docs if is_fewshot else [],
        fewshot_labels=fewshot_labels if is_fewshot else [],
        fewshot_reasons=fewshot_reasons if is_fewshot and use_fewshot_reasons else [],
        use_fewshot_reasons=use_fewshot_reasons,
        lang="en",
        task_name=task_name,
        dataset_name="emotion_lines",
        output_dir_path=Path("experiments/emotion_lines"),
        report_path=Path("experiments/emotion_lines/report.csv"),
        documents=documents,
        labels=labels,
        label_dict=label_dict,
        unwanted_labels=["non-neutral"],
    )
    evaluator.start(report_only=report_only)


@app.command()
def wikisectionen(
    model: ModelsEnum,
    port: int,
    report_only: bool = False,
    is_fewshot: bool = False,
    num_fewshot_examples: int = 2,
    anno_guide_in_user_prompt: bool = False,
    use_fewshot_reasons: bool = False,
    is_fewshot_json: bool = False,
):
    from experiments.wikisection_en.details import (
        label_dict,
        system_prompt_template,
        system_prompt_with_guidelines_template,
        user_prompt_template,
        user_prompt_with_guidelines_template,
        project_name,
        project_details,
    )

    # load dataset
    df = pd.read_parquet(
        "datasets/wikisection/en/city/wikisection_en_city_test.parquet"
    )
    documents: List[List[str]] = [
        list(document_sentences) for document_sentences in df["sentences"].tolist()
    ][:NUM_EXAMPLES]
    labels: List[List[str]] = [list(labels) for labels in df["labels"].tolist()][
        :NUM_EXAMPLES
    ]

    if is_fewshot:
        fewshot_df = pd.read_parquet(
            f"datasets/wikisection/en/city/few_shot_nAll_k{num_fewshot_examples}.parquet"
        )
        fewshot_docs: List[List[str]] = [
            list(document_sentences)
            for document_sentences in fewshot_df["sentences"].tolist()
        ]
        fewshot_labels: List[List[str]] = [
            list(labels) for labels in fewshot_df["labels"].tolist()
        ]
        if use_fewshot_reasons:
            fewshot_reasons: List[List[str]] = [
                list(labels) for labels in fewshot_df[f"reasons_{model}"].tolist()
            ]
        else:
            fewshot_reasons = []

    task_name = "few-shot" if is_fewshot else "zero-shot"
    if is_fewshot:
        task_name += f"-{num_fewshot_examples}"
    if use_fewshot_reasons:
        task_name += "-with-reasons"
    if anno_guide_in_user_prompt:
        task_name += "-with-anno-guide-in-user-prompt"
    if is_fewshot_json:
        task_name += "-is-json"

    # start evaluator
    evaluator = SentenceLLMEvaluator(
        model=model,
        port=port,
        project_name=project_name,
        project_details=project_details,
        system_prompt_template=system_prompt_template,
        system_prompt_with_guidelines_template=system_prompt_with_guidelines_template,
        user_prompt_template=user_prompt_template,
        user_prompt_with_guidelines_template=user_prompt_with_guidelines_template,
        is_anno_guide_in_user_prompt=anno_guide_in_user_prompt,
        is_fewshot=is_fewshot,
        is_fewshot_json=is_fewshot_json,
        fewshot_documents=fewshot_docs if is_fewshot else [],
        fewshot_labels=fewshot_labels if is_fewshot else [],
        fewshot_reasons=fewshot_reasons if is_fewshot and use_fewshot_reasons else [],
        use_fewshot_reasons=use_fewshot_reasons,
        lang="en",
        task_name=task_name,
        dataset_name="wikisectionen",
        output_dir_path=Path("experiments/wikisection_en"),
        report_path=Path("experiments/wikisection_en/report.csv"),
        documents=documents,
        labels=labels,
        label_dict=label_dict,
    )
    evaluator.start(report_only=report_only)


@app.command()
def wikisectionde(
    model: ModelsEnum,
    port: int,
    report_only: bool = False,
    is_fewshot: bool = False,
    num_fewshot_examples: int = 2,
    anno_guide_in_user_prompt: bool = False,
    use_fewshot_reasons: bool = False,
    is_fewshot_json: bool = False,
):
    from experiments.wikisection_de.details import (
        label_dict,
        system_prompt_template,
        system_prompt_with_guidelines_template,
        user_prompt_template,
        user_prompt_with_guidelines_template,
        project_name,
        project_details,
    )

    # load dataset
    df = pd.read_parquet(
        "datasets/wikisection/de/city/wikisection_de_city_test.parquet"
    )
    documents: List[List[str]] = [
        list(document_sentences) for document_sentences in df["sentences"].tolist()
    ][:NUM_EXAMPLES]
    labels: List[List[str]] = [list(labels) for labels in df["labels"].tolist()][
        :NUM_EXAMPLES
    ]

    if is_fewshot:
        fewshot_df = pd.read_parquet(
            f"datasets/wikisection/de/city/few_shot_nAll_k{num_fewshot_examples}.parquet"
        )
        fewshot_docs: List[List[str]] = [
            list(document_sentences)
            for document_sentences in fewshot_df["sentences"].tolist()
        ]
        fewshot_labels: List[List[str]] = [
            list(labels) for labels in fewshot_df["labels"].tolist()
        ]
        if use_fewshot_reasons:
            fewshot_reasons: List[List[str]] = [
                list(labels) for labels in fewshot_df[f"reasons_{model}"].tolist()
            ]
        else:
            fewshot_reasons = []

    task_name = "few-shot" if is_fewshot else "zero-shot"
    if is_fewshot:
        task_name += f"-{num_fewshot_examples}"
    if use_fewshot_reasons:
        task_name += "-with-reasons"
    if anno_guide_in_user_prompt:
        task_name += "-with-anno-guide-in-user-prompt"
    if is_fewshot_json:
        task_name += "-is-json"

    # start evaluator
    evaluator = SentenceLLMEvaluator(
        model=model,
        port=port,
        project_name=project_name,
        project_details=project_details,
        system_prompt_template=system_prompt_template,
        system_prompt_with_guidelines_template=system_prompt_with_guidelines_template,
        user_prompt_template=user_prompt_template,
        user_prompt_with_guidelines_template=user_prompt_with_guidelines_template,
        is_anno_guide_in_user_prompt=anno_guide_in_user_prompt,
        is_fewshot=is_fewshot,
        is_fewshot_json=is_fewshot_json,
        fewshot_documents=fewshot_docs if is_fewshot else [],
        fewshot_labels=fewshot_labels if is_fewshot else [],
        fewshot_reasons=fewshot_reasons if is_fewshot and use_fewshot_reasons else [],
        use_fewshot_reasons=use_fewshot_reasons,
        lang="de",
        task_name=task_name,
        dataset_name="wikisectionde",
        output_dir_path=Path("experiments/wikisection_de"),
        report_path=Path("experiments/wikisection_de/report.csv"),
        documents=documents,
        labels=labels,
        label_dict=label_dict,
    )
    evaluator.start(report_only=report_only)


@app.command()
def semeval23persuasionen(
    model: ModelsEnum,
    port: int,
    report_only: bool = False,
    is_fewshot: bool = False,
    num_fewshot_examples: int = 2,
    anno_guide_in_user_prompt: bool = False,
    use_fewshot_reasons: bool = False,
    is_fewshot_json: bool = False,
):
    from experiments.semeval23persuasion_en.details import (
        label_dict,
        system_prompt_template,
        system_prompt_with_guidelines_template,
        user_prompt_template,
        user_prompt_with_guidelines_template,
        project_name,
        project_details,
    )

    # load dataset
    df = pd.read_parquet(
        "datasets/semeval23persuasion/en/test-labels-subtask-3.parquet"
    )
    documents: List[List[str]] = [
        list(document_sentences) for document_sentences in df["sentences"].tolist()
    ][:NUM_EXAMPLES]
    labels: List[List[str]] = [list(labels) for labels in df["labels"].tolist()][
        :NUM_EXAMPLES
    ]
    # TODO DER TYP STIMMT NICHT

    if is_fewshot:
        fewshot_df = pd.read_parquet(
            f"datasets/semeval23persuasion/en/few_shot_nAll_k{num_fewshot_examples}.parquet"
        )
        fewshot_docs: List[List[str]] = [
            list(document_sentences)
            for document_sentences in fewshot_df["sentences"].tolist()
        ]
        fewshot_labels: List[List[str]] = [
            list(labels) for labels in fewshot_df["labels"].tolist()
        ]
        if use_fewshot_reasons:
            fewshot_reasons: List[List[str]] = [
                list(labels) for labels in fewshot_df[f"reasons_{model}"].tolist()
            ]
        else:
            fewshot_reasons = []
        # TODO DER TYP STIMMT NICHT

    task_name = "few-shot" if is_fewshot else "zero-shot"
    if is_fewshot:
        task_name += f"-{num_fewshot_examples}"
    if use_fewshot_reasons:
        task_name += "-with-reasons"
    if anno_guide_in_user_prompt:
        task_name += "-with-anno-guide-in-user-prompt"
    if is_fewshot_json:
        task_name += "-is-json"

    # start evaluator
    evaluator = SentenceLLMEvaluator(
        model=model,
        port=port,
        project_name=project_name,
        project_details=project_details,
        system_prompt_template=system_prompt_template,
        system_prompt_with_guidelines_template=system_prompt_with_guidelines_template,
        user_prompt_template=user_prompt_template,
        user_prompt_with_guidelines_template=user_prompt_with_guidelines_template,
        is_anno_guide_in_user_prompt=anno_guide_in_user_prompt,
        is_fewshot=is_fewshot,
        is_fewshot_json=is_fewshot_json,
        fewshot_documents=fewshot_docs if is_fewshot else [],
        fewshot_labels=fewshot_labels if is_fewshot else [],
        fewshot_reasons=fewshot_reasons if is_fewshot and use_fewshot_reasons else [],
        use_fewshot_reasons=use_fewshot_reasons,
        lang="en",
        task_name=task_name,
        dataset_name="semeval23persuasionen",
        output_dir_path=Path("experiments/semeval23persuasion_en"),
        report_path=Path("experiments/semeval23persuasion_en/report.csv"),
        documents=documents,
        labels=labels,
        label_dict=label_dict,
    )
    evaluator.start(report_only=report_only)


@app.command()
def semeval23persuasionde(
    model: ModelsEnum,
    port: int,
    report_only: bool = False,
    is_fewshot: bool = False,
    num_fewshot_examples: int = 2,
    anno_guide_in_user_prompt: bool = False,
    use_fewshot_reasons: bool = False,
    is_fewshot_json: bool = False,
):
    from experiments.semeval23persuasion_de.details import (
        label_dict,
        system_prompt_template,
        system_prompt_with_guidelines_template,
        user_prompt_template,
        user_prompt_with_guidelines_template,
        project_name,
        project_details,
    )

    # load dataset
    df = pd.read_parquet(
        "datasets/semeval23persuasion/de/test-labels-subtask-3.parquet"
    )
    documents: List[List[str]] = [
        list(document_sentences) for document_sentences in df["sentences"].tolist()
    ][:NUM_EXAMPLES]
    labels: List[List[str]] = [list(labels) for labels in df["labels"].tolist()][
        :NUM_EXAMPLES
    ]

    if is_fewshot:
        fewshot_df = pd.read_parquet(
            f"datasets/semeval23persuasion/de/few_shot_nAll_k{num_fewshot_examples}.parquet"
        )
        fewshot_docs: List[List[str]] = [
            list(document_sentences)
            for document_sentences in fewshot_df["sentences"].tolist()
        ]
        fewshot_labels: List[List[str]] = [
            list(labels) for labels in fewshot_df["labels"].tolist()
        ]
        if use_fewshot_reasons:
            fewshot_reasons: List[List[str]] = [
                list(labels) for labels in fewshot_df[f"reasons_{model}"].tolist()
            ]
        else:
            fewshot_reasons = []

    task_name = "few-shot" if is_fewshot else "zero-shot"
    if is_fewshot:
        task_name += f"-{num_fewshot_examples}"
    if use_fewshot_reasons:
        task_name += "-with-reasons"
    if anno_guide_in_user_prompt:
        task_name += "-with-anno-guide-in-user-prompt"
    if is_fewshot_json:
        task_name += "-is-json"

    # start evaluator
    evaluator = SentenceLLMEvaluator(
        model=model,
        port=port,
        project_name=project_name,
        project_details=project_details,
        system_prompt_template=system_prompt_template,
        system_prompt_with_guidelines_template=system_prompt_with_guidelines_template,
        user_prompt_template=user_prompt_template,
        user_prompt_with_guidelines_template=user_prompt_with_guidelines_template,
        is_anno_guide_in_user_prompt=anno_guide_in_user_prompt,
        is_fewshot=is_fewshot,
        is_fewshot_json=is_fewshot_json,
        fewshot_documents=fewshot_docs if is_fewshot else [],
        fewshot_labels=fewshot_labels if is_fewshot else [],
        fewshot_reasons=fewshot_reasons if is_fewshot and use_fewshot_reasons else [],
        use_fewshot_reasons=use_fewshot_reasons,
        lang="de",
        task_name=task_name,
        dataset_name="semeval23persuasionde",
        output_dir_path=Path("experiments/semeval23persuasion_de"),
        report_path=Path("experiments/semeval23persuasion_de/report.csv"),
        documents=documents,
        labels=labels,
        label_dict=label_dict,
    )
    evaluator.start(report_only=report_only)


if __name__ == "__main__":
    app()
