import rootutils
root_path = rootutils.find_root(search_from=__file__, indicator=[".git"])
rootutils.set_root(
    path=root_path, # path to the root directory
    project_root_env_var=True, # set the PROJECT_ROOT environment variable to root directory
    dotenv=True, # load environment variables from .env if exists in root directory
    pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
    cwd=True, # change current working directory to the root directory (helps with filepaths)
)

from typing import Optional, TypedDict
import pandas as pd
import mlflow
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pathlib import Path
import typer

from mlflow.models import infer_signature
from models.DatsetInputType import DatasetInputType
from models.SentenceTagger import SentenceTagger
from models.SentenceTaggingDataset import SentenceTaggingDataset
import os


assert "AWS_ACCESS_KEY_ID" in os.environ
assert "AWS_SECRET_ACCESS_KEY" in os.environ
assert "MLFLOW_S3_ENDPOINT_URL" in os.environ
assert "MLFLOW_TRACKING_URI" in os.environ

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

class ModelConfig(TypedDict):
    input_type: DatasetInputType
    embedding_dim: Optional[int]
    embedding_model_name: str
    freeze_embedding_model: Optional[bool]

def do_experiment(
    description: str,
    dataset_name: str,
    data_key: str,
    model_config: ModelConfig,
    num_train_samples: Optional[int] = None,
):
    assert dataset_name in train_paths, f"Training Dataset {dataset_name} not found"
    assert dataset_name in val_paths, f"Validation Dataset {dataset_name} not found"
    assert dataset_name in test_paths, f"Test Dataset {dataset_name} not found"

    # params
    batch_size = 32

    # Load the dataframes
    print(f"Loading data for {dataset_name}")
    train_df = pd.read_parquet(
        Path(train_paths[dataset_name]).with_suffix(".embed.parquet")
    )[:num_train_samples]
    val_df = pd.read_parquet(
        Path(val_paths[dataset_name]).with_suffix(".embed.parquet")
    )
    test_df = pd.read_parquet(
        Path(test_paths[dataset_name]).with_suffix(".embed.parquet")
    )

    # Create datasets
    print(f"Creating datasets for {dataset_name}")
    labels_key = "labels"
    train_dataset = SentenceTaggingDataset(train_df, data_key, labels_key)
    val_dataset = SentenceTaggingDataset(val_df, data_key, labels_key)
    test_dataset = SentenceTaggingDataset(test_df, data_key, labels_key)
    assert (
        train_dataset.tags == val_dataset.tags == test_dataset.tags
    )
    assert train_dataset.tag2id == val_dataset.tag2id == test_dataset.tag2id
    assert train_dataset.id2tag == val_dataset.id2tag == test_dataset.id2tag

    # Create DataLoaders
    print(f"Creating dataloaders for {dataset_name}")
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=SentenceTaggingDataset.collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=SentenceTaggingDataset.collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=SentenceTaggingDataset.collate_fn
    )   

    # Initialize the model
    print("Initializing the model")
    model = SentenceTagger(
        use_lstm=True,
        id2tag=train_dataset.id2tag,
        **model_config,
    )

    print("Starting the experiment")
    mlflow.set_experiment("seq-sent-class-new")
    with mlflow.start_run(
        description=description,
        tags={"dataset": dataset_name, 
              "model_name": model_config["embedding_model_name"], 
              "num_train_samples": f"{num_train_samples}" if num_train_samples else "all"},
    ) as run:
        # Setup Logger
        logger = MLFlowLogger(
            experiment_name=mlflow.get_experiment(run.info.experiment_id).name, 
            tracking_uri=mlflow.get_tracking_uri(),
            run_id=run.info.run_id,
            log_model=True,
        )

        # Callbacks
        early_stopping = EarlyStopping(
            monitor="val_avg_loss", patience=3
        )  # Stop if val_loss doesn't improve for 3 epochs

        checkpoint = ModelCheckpoint(
            dirpath=f"./experiments/{run.info.experiment_id}/{run.info.run_id}",
            filename="{epoch}-{val_avg_loss:.2f}",
            save_top_k=3,
            monitor="val_avg_loss",
            mode="min",
        )

        # Trainer
        trainer = Trainer(
            max_epochs=100,
            callbacks=[early_stopping, checkpoint],
            logger=logger,
            precision=32,  # full precision training
            gradient_clip_val=1.0,  # Gradient clipping
            accelerator="gpu",  # Use GPU
            devices=1,  # Use 1 GPU
        )

        # Train the model
        trainer.fit(model, train_dataloader, val_dataloader)

        # Load the best model
        best_model_path = checkpoint.best_model_path
        model = SentenceTagger.load_from_checkpoint(
            checkpoint_path=best_model_path,
            id2tag=test_dataset.id2tag,
        )

        # Evaluate the model
        trainer = Trainer(
            logger=logger,
            accelerator="gpu",  # Use GPU
            devices=1,  # Use 1 GPU
        )
        trainer.test(model, test_dataloader)

        # Register the best model
        embeddings, _, mask  = next(iter(test_dataloader))
        example_input = {
            "x": embeddings.numpy(),
            "mask": mask.numpy(),
        }
        example_output = model(embeddings, mask=mask)
        mlflow.pytorch.log_model(
            model, 
            artifact_path="best-model",
            signature=infer_signature(
                model_input=example_input,
                model_output=example_output,
            )
        )


@app.command()
def precomputed_embeddings(dataset_name: str, num_train_samples: Optional[int] = None):
    do_experiment(
        description="Sequence Tagger Model with Precomputed Sentence Embeddings",
        dataset_name=dataset_name,
        data_key="query_embedding",
        model_config={
            "input_type": DatasetInputType.EMBEDDINGS,
            "embedding_dim": 4096,
            "embedding_model_name": "nvembed",
            "freeze_embedding_model": False,
        },
        num_train_samples=num_train_samples,
    )

@app.command()
def compute_embeddings(dataset_name: str, model_name: str, freeze_model: bool):
    do_experiment(
        description="Sequence Tagger Model that Computes Sentence Embeddings",
        dataset_name=dataset_name,
        data_key="sentences",
        model_config={
            "input_type": DatasetInputType.SENTENCES,
            "embedding_dim": None,
            "embedding_model_name": model_name,
            "freeze_embedding_model": freeze_model,
        }
    )



if __name__ == "__main__":
    app()
