from typing import Iterable, Sequence

import numpy as np
import pandas as pd


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"

LABEL_CANDIDATES = [
    "label", "y", "target", "class", "y_train", "y_val", "y_test"
]

SEQUENCE_CANDIDATES = [
    "sequence", "seq", "pattern", "cleave_pattern", "negative_pattern",
    "x", "x_train", "x_val", "X_val", "x_test", "test_pattern",
]


def normalize_sequence(sequence: str) -> str:
    """Preserve the positional frame by mapping '-' gaps to ProteinBERT's X."""

    return str(sequence).strip().upper().replace("-", "X")


def find_column(df: pd.DataFrame, candidates: Sequence[str]):
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def labels_to_binary(series: pd.Series, csv_path: str) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        labels = series.astype(int).to_numpy()
    else:
        mapping = {
            "1": 1,
            "0": 0,
            "positive": 1,
            "pos": 1,
            "posi": 1,
            "+": 1,
            "negative": 0,
            "neg": 0,
            "nega": 0,
            "-": 0,
        }
        converted = []
        for value in series:
            key = str(value).strip().lower()
            if key not in mapping:
                raise ValueError(f"Unsupported label {value!r} in {csv_path}")
            converted.append(mapping[key])
        labels = np.asarray(converted, dtype=int)

    unknown = set(np.unique(labels).tolist()) - {0, 1}
    if unknown:
        raise ValueError(
            f"Labels must be binary 0/1 in {csv_path}. Found: {sorted(unknown)}"
        )
    return labels


def sliding_windows(
    full_sequence: str,
    window_size: int,
    *,
    include_last: bool = False,
) -> list[str]:
    """Return fixed-size windows.

    include_last=False intentionally preserves the original script's
    range(len(fullseq) - trim_num) behavior.
    """

    stop = len(full_sequence) - window_size + (1 if include_last else 0)
    if stop <= 0:
        return []
    return [full_sequence[i : i + window_size] for i in range(stop)]


def validate_sequences(sequences: Iterable[str], expected_length: int, source_name: str) -> None:
    bad_lengths = [
        (row_no, len(seq), seq)
        for row_no, seq in enumerate(sequences)
        if len(seq) != expected_length
    ]
    if bad_lengths:
        raise ValueError(
            "Sequence length mismatch after gap->X conversion in {}. "
            "Expected {} characters. First mismatches: {}".format(
                source_name, expected_length, bad_lengths[:10]
            )
        )

    invalid = []
    allowed = set(AMINO_ACIDS)
    for row_no, seq in enumerate(sequences):
        if not seq:
            invalid.append((row_no, "empty sequence"))
            continue
        bad_chars = sorted(set(seq) - allowed)
        if bad_chars:
            invalid.append(
                (row_no, "invalid amino acid(s): {}".format("".join(bad_chars)))
            )

    if invalid:
        raise ValueError(
            f"Invalid sequence(s) in {source_name}. First problems: {invalid[:10]}"
        )
