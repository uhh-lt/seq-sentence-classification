import rootutils

from metrics.compute_metrics import compute_metrics
from models.SentenceClassificationDataset import SentenceClassificationDataset
from models.SentenceClassificationFlatDataset import SentenceClassificationFlatDataset

root_path = rootutils.find_root(search_from=__file__, indicator=[".git"])
rootutils.set_root(
    path=root_path,  # path to the root directory
    project_root_env_var=True,  # set the PROJECT_ROOT environment variable to root directory
    dotenv=True,  # load environment variables from .env if exists in root directory
    pythonpath=True,  # add root directory to the PYTHONPATH (helps with imports)
    cwd=True,  # change current working directory to the root directory (helps with filepaths)
)

from sentence_transformers import losses

from typing import List
import pandas as pd
import mlflow
from torch.utils.data import DataLoader
from pathlib import Path
import typer
from setfit import sample_dataset, Trainer, TrainingArguments, SetFitModel
from datasets import Dataset


import os


assert "AWS_ACCESS_KEY_ID" in os.environ
assert "AWS_SECRET_ACCESS_KEY" in os.environ
assert "MLFLOW_S3_ENDPOINT_URL" in os.environ
assert "MLFLOW_TRACKING_URI" in os.environ
assert "CUDA_VISIBLE_DEVICES" in os.environ

app = typer.Typer()


train_paths = {
    "coarsediscourse": "./datasets/coarsediscourse/coursediscourse_train.parquet",
    "csabstruct": "./datasets/csabstruct/train.parquet",
    "dailydialog": "./datasets/daily_dialog/dailydialog_train.parquet",
    "emotionlines": "./datasets/emotion_lines/friends_train.parquet",
    "pubmed200k": "./datasets/pubmed200k/train.parquet",
}

val_paths = {
    "coarsediscourse": "./datasets/coarsediscourse/coursediscourse_test.parquet",
    "csabstruct": "./datasets/csabstruct/validation.parquet",
    "dailydialog": "./datasets/daily_dialog/dailydialog_valid.parquet",
    "emotionlines": "./datasets/emotion_lines/friends_dev.parquet",
    "pubmed200k": "./datasets/pubmed200k/dev.parquet",
}

test_paths = {
    "coarsediscourse": "./datasets/coarsediscourse/coursediscourse_test.parquet",
    "csabstruct": "./datasets/csabstruct/test.parquet",
    "dailydialog": "./datasets/daily_dialog/dailydialog_test.parquet",
    "emotionlines": "./datasets/emotion_lines/friends_test.parquet",
    "pubmed200k": "./datasets/pubmed200k/test.parquet",
}


def do_experiment(
    description: str, dataset_name: str, data_key: str, examples_per_class: int
):
    assert dataset_name in train_paths, f"Training Dataset {dataset_name} not found"
    assert dataset_name in val_paths, f"Validation Dataset {dataset_name} not found"
    assert dataset_name in test_paths, f"Test Dataset {dataset_name} not found"

    # Load the dataframes
    print(f"Loading data for {dataset_name}")
    train_df = pd.read_parquet(
        Path(train_paths[dataset_name]).with_suffix(".embed.parquet")
    )
    val_df = pd.read_parquet(
        Path(val_paths[dataset_name]).with_suffix(".embed.parquet")
    )

    # Create datasets
    print(f"Creating datasets for {dataset_name}")
    labels_key = "labels"

    train_dataset = SentenceClassificationFlatDataset(train_df, data_key, labels_key)
    train_dataset_for_setfit = Dataset.from_dict(
        {
            "text": train_dataset.datas,
            "label": train_dataset.labels,
        }
    )
    train_dataset_for_setfit = sample_dataset(
        train_dataset_for_setfit, label_column="label", num_samples=examples_per_class
    )
    print(train_dataset_for_setfit)

    val_dataset = SentenceClassificationFlatDataset(val_df, data_key, labels_key)
    val_dataset_for_setfit = Dataset.from_dict(
        {
            "text": val_dataset.datas,
            "label": val_dataset.labels,
        }
    )

    test_df = pd.read_parquet(
        Path(test_paths[dataset_name]).with_suffix(".embed.parquet")
    )
    test_dataset = SentenceClassificationDataset(test_df, data_key, labels_key)
    test_dataloader = DataLoader(test_dataset, batch_size=None)

    assert train_dataset.tags == val_dataset.tags == test_dataset.tags
    assert train_dataset.tag2id == val_dataset.tag2id == test_dataset.tag2id
    assert train_dataset.id2tag == val_dataset.id2tag == test_dataset.id2tag

    # Init model
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model = SetFitModel.from_pretrained(
        model_name,
        use_differentiable_head=True,
        head_params={"out_features": len(train_dataset.tag2id)},
    )
    print(model)
    print("Model initialized")

    # Prepare trainer
    args = TrainingArguments(
        batch_size=(
            32,
            2,
        ),  # Set the batch sizes for the embedding and classifier training phases respectively, or set both if an integer is provided.
        num_epochs=(
            1,
            16,
        ),  # Set the number of epochs the embedding and classifier training phases respectively, or set both if an integer is provided.
        sampling_strategy="oversampling",  # Draws even number of positive/ negative sentence pairs until every sentence pair has been drawn.
        body_learning_rate=(
            2e-5,
            1e-5,
        ),  # Set the learning rate for the SentenceTransformer body for the embedding and classifier training phases respectively, or set both if a float is provided
        head_learning_rate=1e-2,  # Set the learning rate for the head for the classifier training phase. Only used with a differentiable PyTorch head.
        loss=losses.CosineSimilarityLoss,  # The loss function to use for contrastive training of the embedding training phase
        distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,  # Function that returns a distance between two embeddings. It is set for the triplet loss and ignored for CosineSimilarityLoss and SupConLoss
        margin=0.25,  # Margin for the triplet loss. Negative samples should be at least margin further apart from the anchor than the positive. It is ignored for CosineSimilarityLoss, BatchHardSoftMarginTripletLoss and SupConLoss.
        end_to_end=False,  # If True, train the entire model end-to-end during the classifier training phase. Otherwise, freeze the SentenceTransformer body and only train the head. Only used with a differentiable PyTorch head.
        warmup_proportion=0.1,  # Proportion of the warmup in the total training steps.
        samples_per_label=2,  # Number of consecutive, random and unique samples drawn per label. This is only relevant for triplet loss and ignored for CosineSimilarityLoss. Batch size should be a multiple of samples_per_label.
        seed=42,  # Random seed that will be set at the beginning of training
        report_to="mlflow",
        logging_strategy="epoch",
        eval_strategy="no",
        save_strategy="epoch",
        load_best_model_at_end=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset_for_setfit,
        eval_dataset=val_dataset_for_setfit,
        metric="accuracy",
    )

    print("Starting the experiment")
    mlflow.set_experiment("setfit2")
    with mlflow.start_run(
        description=description,
        tags={
            "dataset": dataset_name,
            "model_name": model_name,
            "examples_per_class": str(examples_per_class),
        },
    ):
        # Train the model
        trainer.train()
        print("Training complete")

        # Save and load the best model
        trainer.model.save_pretrained("best_model")
        model = SetFitModel.from_pretrained("best_model")
        print("Best model saved and loaded")

        # Evaluate the best model
        trainer.evaluate(metric_key_prefix="val")
        print("Evaluation complete")

        golds: List[List[int]] = []
        preds: List[List[int]] = []
        for batch in test_dataloader:
            data, tags = batch
            predictions = model.predict(data, use_labels=False).tolist()  # type: ignore

            golds.append(tags.tolist())
            preds.append(predictions)  # type: ignore
        print("Test complete")

        precision, recall, f1, acc = compute_metrics(
            id2tag=test_dataset.id2tag,
            preds=preds,
            golds=golds,
        )

        # Log metrics
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_acc", acc)


# @app.command()
# def precomputed_embeddings(dataset_name: str):
#     do_experiment(
#         description="Sequence Tagger Model with Precomputed Sentence Embeddings",
#         dataset_name=dataset_name,
#         data_key="query_embedding",
#         model_config={
#             "input_type": DatasetInputType.EMBEDDINGS,
#             "embedding_dim": 4096,
#             "embedding_model_name": "nvembed",
#             "freeze_embedding_model": False,
#         },
#     )


# @app.command()
# def compute_embeddings(dataset_name: str, model_name: str, freeze_model: bool):
#     do_experiment(
#         description="Sequence Tagger Model that Computes Sentence Embeddings",
#         dataset_name=dataset_name,
#         data_key="sentences",
#         model_config={
#             "input_type": DatasetInputType.SENTENCES,
#             "embedding_dim": None,
#             "embedding_model_name": model_name,
#             "freeze_embedding_model": freeze_model,
#         },
#     )


do_experiment(
    description="Testing stuff",
    data_key="sentences",
    dataset_name="csabstruct",
    examples_per_class=32,
)


# if __name__ == "__main__":
# app()
