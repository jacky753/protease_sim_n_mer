from typing import Iterable

from .contracts import SequenceFeatureExtractor
from .result_writer import ResultWriter
from .sequence_utils import sliding_windows
from .spike_variants import SpikeVariantProvider


class SpikePredictionService:
    """Predict cleavage probability over sliding Spike-protein windows."""

    def __init__(
        self,
        feature_extractor: SequenceFeatureExtractor,
        variant_provider: SpikeVariantProvider,
        writer: ResultWriter,
        *,
        window_size: int = 160,
        include_last_window: bool = False,
    ):
        self.feature_extractor = feature_extractor
        self.variant_provider = variant_provider
        self.writer = writer
        self.window_size = window_size
        self.include_last_window = include_last_window

    def predict_variants(self, classifier, variant_names: Iterable[str]) -> None:
        for mutation_name in variant_names:
            print("#" * 10 + f" Spike prediction: {mutation_name} " + "#" * 10)
            full_sequence = self.variant_provider.get(mutation_name)
            windows = sliding_windows(
                full_sequence,
                self.window_size,
                include_last=self.include_last_window,
            )
            if not windows:
                raise ValueError(
                    f"No {self.window_size}-mer window can be generated for {mutation_name}."
                )

            x_test, normalized_windows = self.feature_extractor.encode(
                windows,
                source_name=f"Spike variant {mutation_name}",
            )
            probabilities = classifier.predict_proba(x_test)[:, 1]
            predicted_classes = classifier.predict(x_test)

            print("predictdata = P(label=1)")
            print(probabilities.shape)
            self.writer.save_spike_prediction(
                mutation_name,
                probabilities,
                predicted_classes,
                normalized_windows,
            )
