from pathlib import Path
from typing import Sequence

import numpy as np

from .config import AppConfig
from .contracts import ClassifierFactory
from .dataset_loader import PreSplitCsvLoader
from .evaluation import BinaryClassificationEvaluator
from .result_writer import ResultWriter


class TrainingPipeline:
    """Coordinate CV, held-out evaluation and final training.

    The pipeline depends on abstractions for model creation and feature-backed
    dataset loading rather than constructing concrete ML components internally.
    """

    def __init__(
        self,
        config: AppConfig,
        dataset_loader: PreSplitCsvLoader,
        classifier_factory: ClassifierFactory,
        evaluator: BinaryClassificationEvaluator,
        writer: ResultWriter,
    ):
        self.config = config
        self.dataset_loader = dataset_loader
        self.classifier_factory = classifier_factory
        self.evaluator = evaluator
        self.writer = writer

    def run(self):
        self._validate_dataset_directory()
        cv_thresholds = self.run_cross_validation()
        self.run_held_out_test(cv_thresholds)
        return self.train_final_model()

    def run_cross_validation(self) -> list[float]:
        roc_auc_values: list[float] = []
        optimal_thresholds: list[float] = []
        protease = self.config.target_protease

        for fold_no in range(1, self.config.cv_folds + 1):
            print("#" * 10 + f" fixed {self.config.cv_folds}-CV fold {fold_no} " + "#" * 10)
            train_csv = self._dataset_path(f"{protease}.fold{fold_no}.train.csv")
            val_csv = self._dataset_path(f"{protease}.fold{fold_no}.val.csv")

            train = self.dataset_loader.load(train_csv)
            val = self.dataset_loader.load(val_csv)
            self.writer.save_cv_trace(
                fold_no,
                train.sequences,
                train.y,
                val.sequences,
                val.y,
            )

            classifier = self.classifier_factory.create()
            classifier.fit(train.x, train.y)
            y_score = classifier.predict_proba(val.x)[:, 1]
            result = self.evaluator.evaluate(val.y, y_score)

            roc_auc_values.append(result.metrics.roc_auc)
            optimal_thresholds.append(result.classification_threshold)
            self.writer.save_cv_fold(fold_no, result)

        self.writer.save_cv_summary(roc_auc_values, optimal_thresholds)
        return optimal_thresholds

    def run_held_out_test(self, cv_thresholds: Sequence[float]) -> None:
        if not cv_thresholds:
            raise ValueError("cv_thresholds must not be empty")

        print("#" * 10 + " Start fixed test validation " + "#" * 10)
        protease = self.config.target_protease
        train = self.dataset_loader.load(self._dataset_path(f"{protease}.train.csv"))
        test = self.dataset_loader.load(self._dataset_path(f"{protease}.test.csv"))
        self.writer.save_test_trace(
            train.sequences,
            train.y,
            test.sequences,
            test.y,
        )

        classifier = self.classifier_factory.create()
        classifier.fit(train.x, train.y)
        y_score = classifier.predict_proba(test.x)[:, 1]

        # Preserve the original anti-leakage rule: never optimize on test data.
        test_threshold = float(np.mean(cv_thresholds))
        result = self.evaluator.evaluate(
            test.y,
            y_score,
            classification_threshold=test_threshold,
        )
        self.writer.save_test(result)

    def train_final_model(self):
        protease = self.config.target_protease
        dataset = self.dataset_loader.load(
            self._dataset_path(f"{protease}.dataset.csv")
        )
        classifier = self.classifier_factory.create()
        classifier.fit(dataset.x, dataset.y)
        print(
            "Final model trained with:",
            f"{protease}.dataset.csv",
            dataset.x.shape,
        )
        return classifier

    def _dataset_path(self, filename: str) -> Path:
        return self.config.split_dataset_dir / filename

    def _validate_dataset_directory(self) -> None:
        print("split_dataset_dir:", self.config.split_dataset_dir)
        if not self.config.split_dataset_dir.is_dir():
            raise FileNotFoundError(
                f"Pre-split dataset directory was not found: {self.config.split_dataset_dir}"
            )
