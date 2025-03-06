import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SentenceClassificationFlatDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, data_key: str, labels_key: str):
        assert data_key in dataframe.columns, f"{data_key} not in dataframe"
        assert labels_key in dataframe.columns, f"{labels_key} not in dataframe"
        assert dataframe[data_key].apply(lambda x: isinstance(x, np.ndarray)).all(), "Data colum must be numpy arrays"

        datas = []
        tags = []
        unique_tags = set()
        for idx, row in dataframe.iterrows():
            data = list(row[data_key])
            labels = list(row[labels_key])

            unique_tags.update(labels)

            datas.extend(data)
            tags.extend(labels)

        self.tags = sorted(list(unique_tags))
        self.tag2id = {tag: i for i, tag in enumerate(self.tags)}
        self.id2tag = {i: tag for tag, i in self.tag2id.items()}

        self.datas = datas
        self.labels = [self.tag2id[tag] for tag in tags]

        assert len(self.datas) == len(self.labels)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.datas[idx]
        label = self.labels[idx]
        return data, label


    @staticmethod
    def collate_fn(batch):
        data, labels = zip(*batch)

        return (
            list(data) if isinstance(data[0], str) else torch.tensor(data),
            torch.tensor(labels),
        )

# train_df = pd.read_parquet("./datasets/csabstruct/train.embed.parquet")


# data_key = "sentences"
# labels_key = "labels"
# dataset = SentenceClassificationDataset(train_df, data_key, labels_key)
# dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# for batch in dataloader:
#     sentences, labels = batch
#     print("Sentences:", sentences)
#     print("Labels:", labels)
#     print("Batch Size (sentences):", len(sentences))  # Will vary
#     print()
