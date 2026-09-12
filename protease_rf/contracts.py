from typing import Protocol, Sequence, Tuple, Any

import numpy as np


class SequenceFeatureExtractor(Protocol):
    """Small interface required by dataset loading and Spike prediction."""

    def encode(
        self,
        sequences: Sequence[str],
        *,
        source_name: str = "sequences",
    ) -> Tuple[np.ndarray, list[str]]:
        ...


class ClassifierFactory(Protocol):
    """Factory interface so the pipeline is not tied to RandomForest."""

    def create(self) -> Any:
        ...
