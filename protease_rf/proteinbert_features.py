from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from proteinbert.model_generation import tokenize_seq

from .sequence_utils import normalize_sequence, validate_sequences


class ProteinBertFeatureExtractor:
    """Extract flattened local ProteinBERT representations.

    This class owns only ProteinBERT concerns: model construction, tokenization,
    inference and feature caching. Dataset CSV rules live elsewhere.
    """

    def __init__(self, expected_raw_length: int = 160, batch_size: int = 1):
        self.expected_raw_length = expected_raw_length
        self.batch_size = batch_size
        self._model_generator, self._input_encoder = load_pretrained_model()
        self._model_cache: dict[int, object] = {}
        self._feature_cache: dict[str, np.ndarray] = {}
        self._assert_x_supported()

    @staticmethod
    def _assert_x_supported() -> None:
        try:
            tokenize_seq("X")
        except Exception as exc:
            raise RuntimeError(
                "This ProteinBERT tokenizer does not accept 'X'. Gap positions "
                "are converted from '-' to 'X' to preserve 160-mer coordinates. "
                f"Original error: {exc}"
            ) from exc

    def encode(
        self,
        sequences: Sequence[str],
        *,
        source_name: str = "sequences",
    ) -> tuple[np.ndarray, list[str]]:
        if len(sequences) == 0:
            raise ValueError(f"No sequences found in {source_name}")

        normalized = [normalize_sequence(seq) for seq in sequences]
        validate_sequences(normalized, self.expected_raw_length, source_name)

        unseen_by_length: dict[int, list[str]] = defaultdict(list)
        for seq in normalized:
            if seq in self._feature_cache:
                continue
            bucket = unseen_by_length[len(seq)]
            if seq not in bucket:
                bucket.append(seq)

        for raw_len, seq_list in sorted(unseen_by_length.items()):
            model = self._model_for_raw_length(raw_len)
            seq_series = pd.Series(seq_list, dtype="object")
            encoded_seq, encoded_annt = self._input_encoder.encode_X(
                seq_series, raw_len + 2
            )

            # Keep one-row inference from the original implementation to control
            # memory use and to preserve the same local representation path.
            for local_no, seq in enumerate(seq_list):
                local_rep, _ = model.predict(
                    [
                        encoded_seq[local_no : local_no + 1],
                        encoded_annt[local_no : local_no + 1],
                    ],
                    batch_size=self.batch_size,
                    verbose=0,
                )
                self._feature_cache[seq] = (
                    local_rep[0].astype(np.float32).flatten()
                )

        features = [self._feature_cache[seq] for seq in normalized]
        return np.vstack(features).astype(np.float32), normalized

    def _model_for_raw_length(self, raw_len: int):
        seq_len = raw_len + 2
        if seq_len not in self._model_cache:
            print(f"Create ProteinBERT model for seq_len={seq_len}")
            model = self._model_generator.create_model(seq_len)
            self._model_cache[seq_len] = get_model_with_hidden_layers_as_outputs(model)
        return self._model_cache[seq_len]
