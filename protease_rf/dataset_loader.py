from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import SequenceFeatureExtractor
from .sequence_utils import (
    LABEL_CANDIDATES,
    SEQUENCE_CANDIDATES,
    find_column,
    labels_to_binary,
)


@dataclass
class EncodedDataset:
    x: np.ndarray
    y: np.ndarray
    sequences: list[str]
    raw: pd.DataFrame


class PreSplitCsvLoader:
    """Load one pre-split CSV and delegate sequence encoding."""

    def __init__(self, feature_extractor: SequenceFeatureExtractor):
        self.feature_extractor = feature_extractor

    def load(self, csv_path: Path) -> EncodedDataset:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Pre-split CSV was not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]

        label_col = find_column(df, LABEL_CANDIDATES)
        seq_col = find_column(df, SEQUENCE_CANDIDATES)

        if label_col is None:
            label_col = self._find_binary_column(df)
        if label_col is None:
            raise ValueError(
                f"Could not detect the label column in {csv_path}. columns={list(df.columns)}"
            )
        if seq_col is None:
            raise ValueError(
                f"Could not detect the sequence column in {csv_path}. columns={list(df.columns)}"
            )

        y = labels_to_binary(df[label_col], str(csv_path))
        sequence_rows = df[seq_col].astype(str).tolist()
        x, normalized_sequences = self.feature_extractor.encode(
            sequence_rows,
            source_name=str(csv_path),
        )

        if len(x) != len(y):
            raise ValueError(
                f"Feature/label row mismatch in {csv_path}: X={len(x)}, y={len(y)}"
            )

        print(
            "loaded+encoded:",
            csv_path.name,
            "X:",
            x.shape,
            "y:",
            y.shape,
            "class counts:",
            dict(zip(*np.unique(y, return_counts=True))),
        )
        return EncodedDataset(x=x, y=y, sequences=normalized_sequences, raw=df)

    @staticmethod
    def _find_binary_column(df: pd.DataFrame):
        for col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna().unique()
            if len(values) > 0 and set(values.tolist()).issubset({0, 1}):
                return col
        return None
