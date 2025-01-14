import torch
from torch import nn
from pytorch_lightning import LightningModule
from sentence_transformers import SentenceTransformer
from torchcrf import CRF
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class SentenceTagger(LightningModule):
    def __init__(
        self,
        sentence_model_name,
        num_tags,
        hidden_dim=256,
        use_lstm=True,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for easy loading

        self.sentence_model = SentenceTransformer(sentence_model_name)
        self.embedding_dim = self.sentence_model.get_sentence_embedding_dimension()
        assert self.embedding_dim is not None, "Model must return fixed-size embeddings"

        if use_lstm:
            self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, batch_first=True)
        else:
            self.lstm = None

        self.linear = nn.Linear(
            hidden_dim if use_lstm else self.embedding_dim, num_tags
        )
        self.crf = CRF(num_tags)

        self.learning_rate = learning_rate

    def forward(self, x, tags=None, mask=None):
        x = self.sentence_model.encode(x, convert_to_tensor=True)

        if self.lstm:
            x, _ = self.lstm(x)

        emissions = self.linear(x)

        if tags is None:
            return self.crf.decode(emissions, mask=mask)
        else:
            return -self.crf(emissions, tags, mask=mask)  # Negative log-likelihood loss

    def training_step(self, batch, batch_idx):
        sentences, tags, mask = batch
        loss = self(sentences, tags=tags, mask=mask)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sentences, tags, mask = batch
        loss = self(sentences, tags=tags, mask=mask)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


class SentenceTaggingDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

        unique_tags = set()
        for idx, row in self.dataframe.iterrows():
            labels = list(row["labels"])
            unique_tags.update(labels)
        self.tags = list(unique_tags)
        self.tag2id = {tag: i + 1 for i, tag in enumerate(self.tags)}

        # create a new row "tags" that uses the ids, not the strings
        self.dataframe["tags"] = self.dataframe["labels"].apply(
            lambda x: [self.tag2id[tag] for tag in x]
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        sentences = row["sentences"].tolist()
        labels = row["tags"]
        return sentences, torch.tensor(labels)


def collate_fn(batch):
    sentences, labels = zip(*batch)

    # Pad labels
    labels = [torch.tensor(l) for l in labels]
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

    # Create mask
    mask = torch.zeros(padded_labels.shape, dtype=torch.bool)
    for i, l in enumerate(labels):
        mask[i, : len(l)] = 1

    # Pad sentences (updated)
    max_length = max(len(s) for s in sentences)
    padded_sentences = [s + [""] * (max_length - len(s)) for s in sentences]

    return padded_sentences, padded_labels, mask


def main():
    # Load the dataframes
    train_df = pd.read_parquet(
        "/home/tfischer/Development/seq-sentence-classification/datasets/csabstruct/train.parquet"
    )
    val_df = pd.read_parquet(
        "/home/tfischer/Development/seq-sentence-classification/datasets/csabstruct/validation.parquet"
    )

    # Create datasets
    train_dataset = SentenceTaggingDataset(train_df)
    val_dataset = SentenceTaggingDataset(val_df)
    assert (
        train_dataset.tags == val_dataset.tags
    ), "Tags must be the same in train and val"

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)

    # Initialize the model
    model = SentenceTagger(
        sentence_model_name="sentence-transformers/all-mpnet-base-v2",
        num_tags=len(train_dataset.tags),
        use_lstm=True,
    )

    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3
    )  # Stop if val_loss doesn't improve for 3 epochs
    checkpoint = ModelCheckpoint(
        dirpath="/home/tfischer/Development/seq-sentence-classification/experiments/csabstruct/seq-tagger-models",
        filename="best-model",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    # Logger
    # tensorboard --logdir /home/tfischer/Development/seq-sentence-classification/experiments/csabstruct/seq-tagger-logs
    logger = TensorBoardLogger(
        "/home/tfischer/Development/seq-sentence-classification/experiments/csabstruct/seq-tagger-logs",
        name="sentence_tagger",
    )

    # Trainer
    trainer = Trainer(
        max_epochs=10,
        callbacks=[early_stopping, checkpoint],
        logger=logger,
        precision=32,  # full precision training
        gradient_clip_val=1.0,  # Gradient clipping
        accelerator="gpu",  # Use GPU
        devices=1,  # Use 1 GPU
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
