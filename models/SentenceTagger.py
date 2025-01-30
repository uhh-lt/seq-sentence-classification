from enum import Enum
from typing import Dict, List, Optional, TypedDict, Union
from sentence_transformers import SentenceTransformer
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchcrf import CRF
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



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




class SentenceTaggingInputType(str, Enum):
    EMBEDDINGS = "embeddings"
    SENTENCES = "sentences"

class ValidationOutput(TypedDict):
    loss: List[float]
    predictions: List[List[int]]
    tags: List[List[int]]


class SentenceTagger(LightningModule):
    def __init__(
        self,
        id2tag: Dict[int, str],
        hidden_dim=256,
        use_lstm=True,
        learning_rate=1e-3,
        input_type=SentenceTaggingInputType.EMBEDDINGS,
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
            case SentenceTaggingInputType.EMBEDDINGS:
                assert embedding_dim is not None, "embedding_dim must be provided"
                self.embedding_dim = embedding_dim

            case SentenceTaggingInputType.SENTENCES:
                assert embedding_model_name is not None, "embedding_model_name must be provided"
                assert freeze_embedding_model is not None, "freeze_embedding_model must be provided"

                self.sentence_model = SentenceTransformer(embedding_model_name)
                self.embedding_dim = self.sentence_model.get_sentence_embedding_dimension()
                assert self.embedding_dim is not None, "Model must return fixed-size embeddings"

                # freeze the embedding model
                if freeze_embedding_model:
                    for param in self.sentence_model.parameters():
                        param.requires_grad = False

        if use_lstm:
            self.lstm = nn.LSTM(
                self.embedding_dim, hidden_dim, batch_first=True, bidirectional=True
            )
            linear_input_dim = 2 * hidden_dim  # Double the hidden_dim for bidirectional
        else:
            linear_input_dim = self.embedding_dim
            self.lstm = None

        self.linear = nn.Linear(linear_input_dim, len(id2tag))
        self.crf = CRF(len(id2tag), batch_first=True)

        # Set training params
        self.tag2id = id2tag
        self.learning_rate = learning_rate

        # Init outputs
        self.report_path = Path(
            "./experiments/new-seq-tagger-report.csv"
        )
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


    def forward(self, x, tags=None, mask=None):
        assert mask is not None, "Mask must be provided"

        if self.input_type == SentenceTaggingInputType.SENTENCES:
            # x: (batch_size, seq_len, 1)
            assert isinstance(x, List), "x must be a list of list of sentences"
            assert isinstance(x[0], List), "x must be a list of list of sentences"
            assert isinstance(x[0][0], str), "x must be a list of list of sentences"

            # Compute embeddings
            embeddings = []
            for sentences in x:  # Iterate over the batch
                embeddings.append(
                    self.sentence_model.encode(sentences, convert_to_tensor=True)
                )
            x = torch.stack(embeddings)
        elif self.input_type == SentenceTaggingInputType.EMBEDDINGS:
            # x: (batch_size, seq_len, embedding_dim)
            assert x.dim() == 3, "x must be a 3D tensor"
            assert x.size(2) == self.embedding_dim, "x must have the correct embedding dimension"

        if self.lstm:
            # Calculate lengths of valid sequences
            lengths = mask.sum(dim=1).tolist()  

            # Pack embeddings:
            packed_embeddings = pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            
            )

            # Pass packed sequence to LSTM
            packed_output, _ = self.lstm(
                packed_embeddings
            )  

             # Unpack the output
            x, _ = pad_packed_sequence(
                packed_output, batch_first=True
            ) 

        emissions = self.linear(x)

        if tags is None:
            # tags are not provided -> inference -> return predictions
            return self.crf.decode(emissions, mask=mask)
        else:
            # tags are provided -> training -> calculate loss
            return -self.crf(emissions, tags, mask=mask)  # Negative log-likelihood loss

    def training_step(self, batch, batch_idx):
        sent_embs, tags, mask = batch
        loss = self(sent_embs, tags=tags, mask=mask)
        self.log("train_loss", loss)
        return loss

    def val_test_step(self, batch, batch_idx):
        sent_embs, tags, mask = batch

        # Compute validation loss
        loss = self(sent_embs, tags=tags, mask=mask)
        self.log("val_loss", loss)

        # Compute predictions
        preds = self(sent_embs, mask=mask)

        # Extract ground truth tags
        golds = []
        for i in range(len(tags)):  # Iterate over the batch
            golds.append(tags[i][mask[i] == 1].tolist())

        return loss, preds, golds

    def validation_step(self, batch, batch_idx):
        loss, preds, golds = self.val_test_step(batch, batch_idx)

        # log outputs
        self.log("val_loss", loss)
        self.validation_outputs["loss"].append(loss.item())
        self.validation_outputs["predictions"].extend(preds)
        self.validation_outputs["tags"].extend(golds)

        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, golds = self.val_test_step(batch, batch_idx)

        # log outputs
        self.log("test_loss", loss)
        self.test_outputs["loss"].append(loss.item())
        self.test_outputs["predictions"].extend(preds)
        self.test_outputs["tags"].extend(golds)

        return loss

    def compute_metrics(self, outputs: ValidationOutput):
        # convert predictions and golds to BIO format
        preds = [
            [self.tag2id[p] for p in predictions]
            for predictions in outputs["predictions"]
        ]
        golds = [[self.tag2id[t] for t in tags] for tags in outputs["tags"]]
        preds = to_bio_format(preds)
        golds = to_bio_format(golds)

        # compute metrics
        avg_loss = torch.tensor(outputs["loss"]).mean()
        acc = accuracy_score(golds, preds)

        report = classification_report(golds, preds, output_dict=True)
        df_report = pd.DataFrame(report)
        precision, recall, f1, support = [
            round(x * 100.0, 2) for x in df_report["weighted avg"].tolist()
        ]        

        return avg_loss, precision, recall, f1, acc

    def on_validation_epoch_end(self) -> None:
        avg_loss, precision, recall, f1, acc = self.compute_metrics(self.validation_outputs)

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
        avg_loss, precision, recall, f1, acc = self.compute_metrics(self.test_outputs)

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
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def _add_results_to_report(self, results: Dict[str, Union[str, float]]):
        # Access the logger and experiment name
        logger = self.logger
        if isinstance(logger, TensorBoardLogger):
            experiment_name = logger.version
        else:
            experiment_name = "unknown"

        # read existing report, or create new one
        if self.report_path.exists():
            df_current = pd.read_csv(self.report_path)
        else:
            df_current = pd.DataFrame(columns=["Run", "Model"])

        df_new = pd.DataFrame(
            {
                "Run": [experiment_name],
                **{k: [v] for k, v in results.items()},
            }
        )
        df = pd.concat([df_current, df_new])
        df.to_csv(self.report_path, index=False)