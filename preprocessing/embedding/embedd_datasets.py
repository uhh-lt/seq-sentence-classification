import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import tempfile
import torch
import os
from pathlib import Path


def save_embeddings_to_tempfile(embeddings_list, prefix):
    temp_file = tempfile.NamedTemporaryFile(delete=False, prefix=prefix, suffix=".pt")
    torch.save(embeddings_list, temp_file.name)
    return temp_file.name


def load_embeddings_from_tempfiles(temp_files):
    embeddings_list = []
    for temp_file in temp_files:
        embeddings_list.extend(torch.load(temp_file))

    return [embedding.tolist() for embedding in embeddings_list]


def remove_all_tempfiles(temp_files):
    for temp_file in temp_files:
        os.remove(temp_file)


train_paths = {
    "csabstruct": "../datasets/csabstruct/train.parquet",
    "coarsediscourse": "../datasets/coarsediscourse/coursediscourse_train.parquet",
    "dailydialog": "../datasets/daily_dialog/dailydialog_train.parquet",
    "emotionlines": "../datasets/emotion_lines/friends_train.parquet",
    "pubmed200k": "../datasets/pubmed200k/train.parquet",
}

val_paths = {
    "csabstruct": "../datasets/csabstruct/validation.parquet",
    "coarsediscourse": "../datasets/coarsediscourse/coursediscourse_test.parquet",
    "dailydialog": "../datasets/daily_dialog/dailydialog_valid.parquet",
    "emotionlines": "../datasets/emotion_lines/friends_dev.parquet",
    "pubmed200k": "../datasets/pubmed200k/dev.parquet",
}

test_paths = {
    "csabstruct": "../datasets/csabstruct/test.parquet",
    "coarsediscourse": "../datasets/coarsediscourse/coursediscourse_test.parquet",
    "dailydialog": "../datasets/daily_dialog/dailydialog_test.parquet",
    "emotionlines": "../datasets/emotion_lines/friends_test.parquet",
    "pubmed200k": "../datasets/pubmed200k/test.parquet",
}

# Copied from: https://huggingface.co/nvidia/NV-Embed-v2/blob/main/instructions.json
task_name_to_instruct = {"STS": "Retrieve semantically similar text"}
query_prefix = "Instruct: " + task_name_to_instruct["STS"] + "\nQuery: "

# load model with t okenizer
model = SentenceTransformer("nvidia/NV-Embed-v2", trust_remote_code=True)
model.max_seq_length = 32768
model.tokenizer.padding_side = "right"
print("Model loaded")


def add_eos(input_examples):
    input_examples = [
        input_example + model.tokenizer.eos_token for input_example in input_examples
    ]
    return input_examples


for dataset_name in train_paths.keys():
    dataset_paths = {
        "train": train_paths[dataset_name],
        "val": val_paths[dataset_name],
        "test": test_paths[dataset_name],
    }
    dataset = {
        "train": pd.read_parquet(dataset_paths["train"]),
        "val": pd.read_parquet(dataset_paths["val"]),
        "test": pd.read_parquet(dataset_paths["test"]),
    }

    for split in ["test", "train", "val"]:
        # determine if embeddings are already present
        if Path(dataset_paths[split]).with_suffix(".embed.parquet").exists():
            print(
                f"Embeddings already present in dataset {dataset_name}. Skipping embedding."
            )
            continue

        # iterate over the dataset and add the embeddings
        query_temp_files = []
        passage_temp_files = []
        query_embeddings_list = []
        passage_embeddings_list = []
        batch_size = 2
        save_interval = 64

        for i, row in tqdm(
            dataset[split].iterrows(), desc=f"Embedding {dataset_name} {split}"
        ):
            sentences = row["sentences"]

            # get the embeddings
            query_embeddings = model.encode(
                add_eos(sentences),
                batch_size=batch_size,
                prompt=query_prefix,
                normalize_embeddings=False,
                convert_to_tensor=True,
            )

            passage_embeddings = model.encode(
                add_eos(sentences),
                batch_size=batch_size,
                normalize_embeddings=False,
                convert_to_tensor=True,
            )
            # add the embeddings to the dataframe
            query_embeddings_list.append(query_embeddings.cpu())
            passage_embeddings_list.append(passage_embeddings.cpu())

            # Save to temporary files every save_interval iterations
            if (i + 1) % save_interval == 0:
                query_temp_files.append(
                    save_embeddings_to_tempfile(
                        query_embeddings_list, f"{dataset_name}_query"
                    )
                )
                passage_temp_files.append(
                    save_embeddings_to_tempfile(
                        passage_embeddings_list, f"{dataset_name}_passage"
                    )
                )
                query_embeddings_list = []
                passage_embeddings_list = []

        # Save any remaining embeddings
        if query_embeddings_list:
            query_temp_files.append(
                save_embeddings_to_tempfile(
                    query_embeddings_list, f"{dataset_name}_query"
                )
            )
        if passage_embeddings_list:
            passage_temp_files.append(
                save_embeddings_to_tempfile(
                    passage_embeddings_list, f"{dataset_name}_passage"
                )
            )

        # Load all embeddings from temporary files
        query_embeddings_list = load_embeddings_from_tempfiles(query_temp_files)
        passage_embeddings_list = load_embeddings_from_tempfiles(passage_temp_files)

        # Add to the dataset
        dataset[split]["query_embedding"] = query_embeddings_list
        dataset[split]["passage_embedding"] = passage_embeddings_list

        # Save the datasets
        dataset[split].to_parquet(
            Path(dataset_paths[split]).with_suffix(".embed.parquet")
        )

        # Remove temporary files
        remove_all_tempfiles(query_temp_files)
        remove_all_tempfiles(passage_temp_files)
