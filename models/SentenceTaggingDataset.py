from typing import List
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd


class SentenceTaggingDataset(Dataset):
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
        return data, labels
    
    @staticmethod
    def collate_fn(batch):
        data, labels = zip(*batch)

        # determine if data is a list of strings (sentences) or a list of list of floats (embeddings)
        # float -> embeddings
        if isinstance(data[0][0][0], float):
            is_embeddings = True
        # str -> sentences
        elif isinstance(data[0][0][0], str):
            is_embeddings = False
        else:
            raise ValueError(f"Data type {type(data[0][0][0])} not supported")

        # Pad labels
        labels = [torch.tensor(ll) for ll in labels]
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

        # Create mask
        mask = torch.zeros(padded_labels.shape, dtype=torch.bool)
        for i, label in enumerate(labels):
            mask[i, : len(label)] = 1

        # Pad data
        if is_embeddings:
            padded_data = pad_sequence([torch.tensor(x) for x in data], batch_first=True, padding_value=0).tolist()
        else:
            max_length = max(len(s) for s in data)
            padded_data = []
            for s in data:
                padded_s = s + [""] * (max_length - len(s))
                padded_data.append(padded_s)

        # switch first (0) with longest (longest_idx)
        longest_idx = max(range(len(labels)), key=lambda k: len(labels[k]))

        def switch_with_first(data: List, idx: int):
            new_data = data.copy()
            new_data[0] = data[idx]
            new_data[idx] = data[0]
            return new_data

        new_padded_data = switch_with_first(padded_data, longest_idx)
        new_padded_labels = switch_with_first(padded_labels.tolist(), longest_idx)
        new_mask = switch_with_first(mask.tolist(), longest_idx)

        assert (
            len(padded_data)
            == len(padded_labels)
            == len(mask)
            == len(new_padded_data)
            == len(new_padded_labels)
            == len(new_mask)
        ), f"Lengths must match: {len(padded_data)}, {len(padded_labels)}, {len(mask)}, {len(new_padded_data)}, {len(new_padded_labels)}, {len(new_mask)}"

        return (
            torch.tensor(new_padded_data),
            torch.tensor(new_padded_labels),
            torch.tensor(new_mask, dtype=torch.bool),
        )

