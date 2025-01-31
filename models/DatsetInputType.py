from enum import Enum


class DatasetInputType(str, Enum):
    EMBEDDINGS = "embeddings"
    SENTENCES = "sentences"
