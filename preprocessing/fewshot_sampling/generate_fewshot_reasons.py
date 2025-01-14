import pandas as pd
from typing import List
from llm_evaluator import ModelsEnum, model_dict
from pathlib import Path
from typing import Dict
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field

from daily_dialog.details import label_dict as daily_dialog_label_dict
from emotion_lines.details import label_dict as emotion_lines_label_dict
from coarsediscourse.details import label_dict as coarsediscourse_label_dict
from csabstruct.details import label_dict as csabstruct_label_dict
from pubmed200k.details import label_dict as pubmed200k_label_dict


K = 2
port = 19269

train_paths = [
    # Path("../datasets/csabstruct/train.parquet"),
    Path("../datasets/pubmed200k/train.parquet"),
    Path("../datasets/coarsediscourse/coursediscourse_train.parquet"),
    Path("../datasets/daily_dialog/dailydialog_train.parquet"),
    Path("../datasets/emotion_lines/friends_train.parquet"),
    Path("datasets/wikisection/en/city/wikisection_en_city_train.parquet"),
    Path("datasets/wikisection/de/city/wikisection_de_city_train.parquet"),
]

label_dicts = [
    # csabstruct_label_dict,
    pubmed200k_label_dict,
    coarsediscourse_label_dict,
    daily_dialog_label_dict,
    emotion_lines_label_dict,
]

fewshot_paths = [path.with_name(f"few_shot_nAll_k{K}.parquet") for path in train_paths]

client = instructor.from_openai(
    OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)


class TextReason(BaseModel):
    text_id: int = Field(description="The id of the text")
    reason: str = Field(description="The reason for the classification")


system_prompt_template = """
You are a professional annotator specialized in assisting with reasoning behind text classifications.
You strictly adhere to the guidelines and follow the desired output format.

Annotation Guidelines:
<annotation_guidelines>

Output Format:
You MUST answer in this JSON format:
[
    {
        "text_id": 1,
        "reason": "The sentence provides context for the research.",
    },
    {
        "text_id": 2,
        "reason": "The sentence presents the research findings.",
    },
    ...
]
"""

user_prompt_template = """
I will give you a numbered list of classified texts.
You must provide a reasoning for the classification of each sentence with the help of the annotation guideliens.

Document:
{document}

Remember you MUST generate a reasoning for every provided text.
"""

for model in ModelsEnum:
    model_name = model.value

    print("Using model", model_name)

    for fewshot_path, label_dict in zip(fewshot_paths, label_dicts):
        print("Processing", fewshot_path)

        fewshot_df = pd.read_parquet(fewshot_path)
        fewshot_docs: List[List[str]] = [
            list(document_sentences)
            for document_sentences in fewshot_df["sentences"].tolist()
        ]
        fewshot_labels: List[List[str]] = [
            list(labels) for labels in fewshot_df["labels"].tolist()
        ]

        annotation_guidelines = ""
        for label, description in label_dict.items():
            annotation_guidelines += f"{label.lower()}\n"
            assert (
                description.count("\n") == 0
            ), "The description must not contain newlines."
            annotation_guidelines += f"{description}\n\n"
        annotation_guidelines = annotation_guidelines.strip()

        system_prompt = system_prompt_template.replace(
            "<annotation_guidelines>", annotation_guidelines
        ).strip()

        fewshot_reasons = []
        for document, labels in zip(fewshot_docs, fewshot_labels):
            document_str = "\n".join(
                [
                    f"ID: {i+1}, Class: {label}, Text: {sentence}"
                    for i, (sentence, label) in enumerate(zip(document, labels))
                ]
            )

            response = client.chat.completions.create(
                model=model_dict[model],
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt_template.format(
                            document=document_str
                        ).strip(),
                    },
                ],
                response_model=List[TextReason],
            )

            parsed_result: Dict[int, str] = {}
            for resp in response:
                parsed_result[resp.text_id] = resp.reason

            # convert reasons to list
            reasons = [
                parsed_result.get(i + 1, "No reason.") for i in range(len(document))
            ]
            fewshot_reasons.append(reasons)

        fewshot_df[f"reasons_{model_name}"] = fewshot_reasons
        fewshot_df.to_parquet(fewshot_path)
