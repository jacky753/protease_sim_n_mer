from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import EvaluationResult


class ResultWriter:
    """Own all output-file naming, CSV serialization and ROC plotting."""

    def __init__(self, output_dir: Path, protease: str):
        self.output_dir = output_dir
        self.protease = protease
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_cv_trace(
        self,
        fold_no: int,
        train_sequences: Sequence[str],
        y_train: np.ndarray,
        val_sequences: Sequence[str],
        y_val: np.ndarray,
    ) -> None:
        pd.DataFrame({"x_train": train_sequences, "y_train": y_train}).to_csv(
            self.output_dir / f"train-data{self.protease}_cv{fold_no}.csv",
            index=False,
        )
        pd.DataFrame({"X_val": val_sequences, "y_val": y_val}).to_csv(
            self.output_dir / f"val-data{self.protease}_cv{fold_no}.csv",
            index=False,
        )

    def save_cv_fold(self, fold_no: int, result: EvaluationResult) -> None:
        pd.DataFrame(
            {"optimal_threshold": [result.classification_threshold]}
        ).to_csv(
            self.output_dir / f"df_optimal_threshold_{self.protease}_cv{fold_no}.csv",
            index=False,
        )

        self._save_metrics(
            result,
            self.output_dir / f"roc_auc_f1score_{self.protease}_cv{fold_no}.csv",
            include_threshold=False,
        )
        self._save_roc(
            result,
            title=f"ROC_{self.protease}_CV{fold_no}",
            path=self.output_dir / f"roc_curve_{self.protease}_cv{fold_no}.png",
        )

        pd.DataFrame(
            {
                "y_val": result.y_true,
                "predict_prob_label1_val": result.y_score,
            }
        ).to_csv(
            self.output_dir / f"df_val_predicteds_{self.protease}_cv{fold_no}.csv",
            index=False,
        )

        pd.DataFrame(
            [[result.metrics.mae, result.metrics.mse]],
            columns=["val_mae", "val_mse"],
        ).to_csv(
            self.output_dir / f"df_val_error_{self.protease}_cv{fold_no}.csv",
            index=False,
        )

    def save_cv_summary(
        self,
        roc_auc_values: Sequence[float],
        thresholds: Sequence[float],
    ) -> None:
        pd.DataFrame(
            {
                "fold": list(range(1, len(roc_auc_values) + 1)),
                "roc_auc": roc_auc_values,
                "optimal_threshold": thresholds,
            }
        ).to_csv(
            self.output_dir / f"cv_summary_{self.protease}.csv",
            index=False,
        )

    def save_test_trace(
        self,
        train_sequences: Sequence[str],
        y_train: np.ndarray,
        test_sequences: Sequence[str],
        y_test: np.ndarray,
    ) -> None:
        pd.DataFrame({"x_learn": train_sequences, "y_learn": y_train}).to_csv(
            self.output_dir / f"learn-data_{self.protease}_seqs.csv",
            index=False,
        )
        pd.DataFrame({"x_test": test_sequences, "y_test": y_test}).to_csv(
            self.output_dir / f"test-data_{self.protease}_seqs.csv",
            index=False,
        )

    def save_test(self, result: EvaluationResult) -> None:
        pd.DataFrame(
            {
                "optimal_threshold": [result.classification_threshold],
                "threshold_source": ["mean_of_3CV_validation_thresholds"],
            }
        ).to_csv(
            self.output_dir / f"df_optimal_threshold_{self.protease}_test.csv",
            index=False,
        )

        self._save_metrics(
            result,
            self.output_dir / f"roc_auc_f1score_{self.protease}_test.csv",
            include_threshold=True,
        )
        self._save_roc(
            result,
            title=f"ROC_{self.protease}_test",
            path=self.output_dir / f"roc_curve_{self.protease}_test.png",
        )

        pd.DataFrame(
            {
                "y_test": result.y_true,
                "predict_prob_label1_test": result.y_score,
                "predicted_class": result.y_pred,
            }
        ).to_csv(
            self.output_dir / f"df_test_predicteds_{self.protease}_test.csv",
            index=False,
        )

        # Filename intentionally matches the original script.
        pd.DataFrame(
            [[result.metrics.mae, result.metrics.mse]],
            columns=["test_mae", "test_mse"],
        ).to_csv(
            self.output_dir / f"df_val_error_{self.protease}_test.csv",
            index=False,
        )

    def save_spike_prediction(
        self,
        mutation_name: str,
        probabilities: np.ndarray,
        predicted_classes: np.ndarray,
        sequences: Sequence[str],
    ) -> None:
        df_predict = pd.DataFrame(
            {
                "predict": probabilities,
                "predict_label": predicted_classes.astype(int),
            },
            index=list(range(len(probabilities))),
        )

        # The first path is intentionally overwritten for each mutation, exactly
        # like the original script; the mutation-specific file below is retained.
        df_predict.to_csv(self.output_dir / f"predict_sprotein_{self.protease}.csv")

        seq_series = pd.Series(list(sequences), name="test_pattern")
        pd.concat([df_predict, seq_series], axis=1).to_csv(
            self.output_dir
            / f"predict_sprotein_seqs_{self.protease}_{mutation_name}.csv"
        )

    @staticmethod
    def _save_metrics(
        result: EvaluationResult,
        path: Path,
        *,
        include_threshold: bool,
    ) -> None:
        data = {
            "roc_auc": [result.metrics.roc_auc],
            "f1_score": [result.metrics.f1],
            "precision_score": [result.metrics.precision],
            "recall_score": [result.metrics.recall],
            "accuracy_value": [result.metrics.accuracy],
        }
        if include_threshold:
            data["classification_threshold"] = [result.classification_threshold]
        pd.DataFrame(data).to_csv(path, index=False)

    @staticmethod
    def _save_roc(result: EvaluationResult, title: str, path: Path) -> None:
        plt.figure()
        plt.plot(result.fpr, result.tpr, marker="o")
        plt.xlabel("FPR: False positive rate")
        plt.ylabel("TPR: True positive rate")
        plt.title(title)
        plt.grid()
        plt.savefig(path)
        plt.close()
