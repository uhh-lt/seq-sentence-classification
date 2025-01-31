import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class SentenceClassificationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, data_key: str, labels_key: str):
        assert data_key in dataframe.columns, f"{data_key} not in dataframe"
        assert labels_key in dataframe.columns, f"{labels_key} not in dataframe"
        assert dataframe[data_key].apply(lambda x: isinstance(x, np.ndarray)).all(), "Data colum must be numpy arrays"

        self.dataframe = dataframe
        self.data_key = data_key

        unique_tags = set()
        for idx, row in self.dataframe.iterrows():
            labels = list(row[labels_key])
            unique_tags.update(labels)
        self.tags = sorted(list(unique_tags))
        self.tag2id = {tag: i for i, tag in enumerate(self.tags)}
        self.id2tag = {i: tag for tag, i in self.tag2id.items()}

        # create a new row "tags" that uses the ids, not the strings
        self.dataframe["tags"] = self.dataframe[labels_key].apply(
            lambda x: [self.tag2id[tag] for tag in x]
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        labels = row["tags"]
        data = row[self.data_key].tolist()

        return (
            data if isinstance(data[0], str) else torch.tensor(data),
            torch.tensor(labels),
        )



