from typing import Dict, List, Optional, TypedDict, Union
from sentence_transformers import SentenceTransformer
import torch
from torch import nn
from pytorch_lightning import LightningModule
import pandas as pd
from pathlib import Path
from pytorch_lightning.loggers import MLFlowLogger

from metrics.compute_metrics import compute_metrics
from models.DatsetInputType import DatasetInputType


class ValidationOutput(TypedDict):
    loss: List[float]
    predictions: List[List[int]]
    tags: List[List[int]]


class SentenceClassifier(LightningModule):
    def __init__(
        self,
        id2tag: Dict[int, str],
        hidden_dim=256,
        learning_rate=1e-3,
        input_type=DatasetInputType.EMBEDDINGS,
        # used for input_type = EMBEDDINGS
        embedding_dim: Optional[int] = None,
        # used for input_type = SENTENCES
        embedding_model_name: Optional[str] = None,
        freeze_embedding_model: Optional[bool] = None,
    ):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for easy loading

        # Init embedding model
        self.input_type = input_type
        match input_type:
            case DatasetInputType.EMBEDDINGS:
                assert embedding_dim is not None, "embedding_dim must be provided"
                self.embedding_dim = embedding_dim

            case DatasetInputType.SENTENCES:
                assert (
                    embedding_model_name is not None
                ), "embedding_model_name must be provided"
                assert (
                    freeze_embedding_model is not None
                ), "freeze_embedding_model must be provided"

                self.sentence_model = SentenceTransformer(embedding_model_name)
                self.embedding_dim = (
                    self.sentence_model.get_sentence_embedding_dimension()
                )
                assert (
                    self.embedding_dim is not None
                ), "Model must return fixed-size embeddings"

                # freeze the embedding model
                if freeze_embedding_model:
                    for param in self.sentence_model.parameters():
                        param.requires_grad = False

        linear_input_dim = self.embedding_dim
        self.linear = nn.Linear(linear_input_dim, len(id2tag))

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Set training params
        self.id2tag = id2tag
        self.learning_rate = learning_rate

        # Init outputs
        self.report_path = Path("./experiments/sent-class-report.csv")
        self.validation_outputs: ValidationOutput = {
            "loss": [],
            "predictions": [],
            "tags": [],
        }
        self.test_outputs: ValidationOutput = {
            "loss": [],
            "predictions": [],
            "tags": [],
        }

    def forward(self, x):
        if self.input_type == DatasetInputType.SENTENCES:
            # x: (batch_size, the_sentence)
            assert isinstance(x, List), "x must be a list of sentences"
            assert isinstance(x[0], str), "x must be a list of sentences"

            # Compute embeddings
            x = self.sentence_model.encode(x, convert_to_tensor=True)
        elif self.input_type == DatasetInputType.EMBEDDINGS:
            # x: (batch_size, embedding_dim)
            assert x.dim() == 2, "x must be a 2D tensor"
            assert (
                x.size(1) == self.embedding_dim
            ), "x must have the correct embedding dimension"

        logits = self.linear(x)
        return logits

    def training_step(self, batch, batch_idx):
        data, tags = batch
        logits = self(data)
        loss = self.loss_fn(logits, tags)
        self.log("train_loss", loss)
        return loss

    def val_test_step(self, batch, batch_idx):
        data, tags = batch

        # Compute validation loss
        logits = self(data)
        loss = self.loss_fn(logits, tags)

        # Compute predictions
        preds = torch.argmax(logits, dim=1).tolist()

        return loss, preds, tags.tolist()

    def validation_step(self, batch, batch_idx):
        loss, preds, golds = self.val_test_step(batch, batch_idx)

        # log outputs
        self.log("val_loss", loss)
        self.validation_outputs["loss"].append(loss.item())
        self.validation_outputs["predictions"].append(preds)
        self.validation_outputs["tags"].append(golds)

        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, golds = self.val_test_step(batch, batch_idx)

        # log outputs
        self.log("test_loss", loss)
        self.test_outputs["loss"].append(loss.item())
        self.test_outputs["predictions"].append(preds)
        self.test_outputs["tags"].append(golds)

        return loss

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.tensor(self.validation_outputs["loss"]).mean()
        precision, recall, f1, acc = compute_metrics(
            id2tag=self.id2tag,
            preds=self.validation_outputs["predictions"],
            golds=self.validation_outputs["tags"],
        )

        self.log("val_avg_loss", avg_loss)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)
        self.log("val_acc", acc)

        self.validation_outputs: ValidationOutput = {
            "loss": [],
            "predictions": [],
            "tags": [],
        }

    def on_test_epoch_end(self) -> None:
        avg_loss = torch.tensor(self.test_outputs["loss"]).mean()
        precision, recall, f1, acc = compute_metrics(
            id2tag=self.id2tag,
            preds=self.test_outputs["predictions"],
            golds=self.test_outputs["tags"],
        )

        self.log("test_avg_loss", avg_loss)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)
        self.log("test_acc", acc)

        self._add_results_to_report(
            {
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Accuracy": round(acc * 100.0, 2),
            }
        )

        self.test_outputs: ValidationOutput = {
            "loss": [],
            "predictions": [],
            "tags": [],
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def _add_results_to_report(self, results: Dict[str, Union[str, float]]):
        # Access the logger and experiment name
        logger = self.logger
        if isinstance(logger, MLFlowLogger):
            run_id = logger._run_id
        else:
            run_id = "unknown"

        # read existing report, or create new one
        if self.report_path.exists():
            df_current = pd.read_csv(self.report_path)
        else:
            df_current = pd.DataFrame(columns=["Run", "Model"])

        df_new = pd.DataFrame(
            {
                "Run": [run_id],
                **{k: [v] for k, v in results.items()},
            }
        )
        df = pd.concat([df_current, df_new])
        df.to_csv(self.report_path, index=False)
