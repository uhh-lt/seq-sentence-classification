import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
from pathlib import Path
import typer

from models.SentenceTagger import SentenceTagger, SentenceTaggingInputType
from models.SentenceTaggingDataset import SentenceTaggingDataset

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

# embedding_key = "query_embedding"  # "passage_embedding"
@app.command()
def main(dataset_name: str, embedding_key: str):
    assert dataset_name in train_paths, f"Training Dataset {dataset_name} not found"
    assert dataset_name in val_paths, f"Validation Dataset {dataset_name} not found"
    assert dataset_name in test_paths, f"Test Dataset {dataset_name} not found"

    # params
    batch_size = 32
    eval_only = False

    # Load the dataframes
    train_df = pd.read_parquet(
        Path(train_paths[dataset_name]).with_suffix(".embed.parquet")
    )
    val_df = pd.read_parquet(
        Path(val_paths[dataset_name]).with_suffix(".embed.parquet")
    )
    test_df = pd.read_parquet(
        Path(test_paths[dataset_name]).with_suffix(".embed.parquet")
    )

    # Create datasets
    labels_key = "labels"
    train_dataset = SentenceTaggingDataset(train_df, embedding_key, labels_key)
    val_dataset = SentenceTaggingDataset(val_df, embedding_key, labels_key)
    test_dataset = SentenceTaggingDataset(test_df, embedding_key, labels_key)
    assert (
        train_dataset.tags == val_dataset.tags == test_dataset.tags
    )
    assert train_dataset.tag2id == val_dataset.tag2id == test_dataset.tag2id
    assert train_dataset.id2tag == val_dataset.id2tag == test_dataset.id2tag

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=SentenceTaggingDataset.collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=SentenceTaggingDataset.collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=SentenceTaggingDataset.collate_fn
    )

    if not eval_only:
        # Initialize the model
        model = SentenceTagger(
            use_lstm=True,
            id2tag=train_dataset.id2tag,
            embedding_dim=4096,
            input_type=SentenceTaggingInputType.EMBEDDINGS,
        )

        # Generate a unique experiment name (e.g., based on timestamp)
        experiment_name = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Logger
        # tensorboard --logdir=./experiments/_tb-logs
        logger = TensorBoardLogger(
            "./experiments/_tb-logs",
            name=dataset_name,
            version=experiment_name,
        )

        # Callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=3
        )  # Stop if val_loss doesn't improve for 3 epochs
        checkpoint = ModelCheckpoint(
            dirpath=f"./experiments/{dataset_name}/seq-tagger-models/{experiment_name}",
            filename="best-model",
            save_top_k=1,
            monitor="val_loss",
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
    model = SentenceTagger.load_from_checkpoint(
        checkpoint_path=f"./experiments/{dataset_name}/seq-tagger-models/{experiment_name}/best-model.ckpt",
        id2tag=test_dataset.id2tag,
    )

    # Evaluate the model
    trainer = Trainer(
        logger=logger if not eval_only else None,
        accelerator="gpu",  # Use GPU
        devices=1,  # Use 1 GPU
    )
    trainer.test(model, test_dataloader)


# if __name__ == "__main__":
main("csabstruct", "query_embedding")

    # app()
