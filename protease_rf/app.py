from .config import AppConfig
from .dataset_loader import PreSplitCsvLoader
from .evaluation import BinaryClassificationEvaluator
from .proteinbert_features import ProteinBertFeatureExtractor
from .random_forest import RandomForestFactory
from .result_writer import ResultWriter
from .runtime import configure_tensorflow_memory_growth
from .spike_prediction import SpikePredictionService
from .spike_variants import SpikeVariantProvider
from .training_pipeline import TrainingPipeline


def build_and_run(config: AppConfig | None = None) -> None:
    """Composition root: construct concrete dependencies in one place."""

    config = config or AppConfig()
    configure_tensorflow_memory_growth()

    print("protease:", config.target_protease)
    print("protease_turn:", config.protease_turn)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    feature_extractor = ProteinBertFeatureExtractor(
        expected_raw_length=config.sequence_length,
        batch_size=config.proteinbert_batch_size,
    )
    dataset_loader = PreSplitCsvLoader(feature_extractor)
    classifier_factory = RandomForestFactory(
        n_estimators=config.rf_n_estimators,
        random_state=config.random_state,
        n_jobs=config.rf_n_jobs,
    )
    evaluator = BinaryClassificationEvaluator()
    writer = ResultWriter(config.output_dir, config.target_protease)

    training = TrainingPipeline(
        config=config,
        dataset_loader=dataset_loader,
        classifier_factory=classifier_factory,
        evaluator=evaluator,
        writer=writer,
    )
    final_classifier = training.run()

    spike_prediction = SpikePredictionService(
        feature_extractor=feature_extractor,
        variant_provider=SpikeVariantProvider(config.project_root),
        writer=writer,
        window_size=config.sequence_length,
        include_last_window=config.include_last_spike_window,
    )
    spike_prediction.predict_variants(
        final_classifier,
        SpikeVariantProvider.DEFAULT_VARIANTS,
    )
