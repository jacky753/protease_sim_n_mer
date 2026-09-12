from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    roc_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    mae: float
    mse: float


@dataclass(frozen=True)
class EvaluationResult:
    y_true: np.ndarray
    y_score: np.ndarray
    y_pred: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    classification_threshold: float
    metrics: ClassificationMetrics


class BinaryClassificationEvaluator:
    """Pure evaluation logic. It neither trains models nor writes files."""

    def evaluate(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        *,
        classification_threshold: Optional[float] = None,
    ) -> EvaluationResult:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = float(roc_auc_score(y_true, y_score))

        threshold = classification_threshold
        if threshold is None:
            optimal_idx = int(np.argmax(tpr - fpr))
            threshold = float(thresholds[optimal_idx])

        y_pred = (y_score >= threshold).astype(int)
        metrics = ClassificationMetrics(
            roc_auc=roc_auc,
            f1=float(f1_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            accuracy=float(accuracy_score(y_true, y_pred)),
            mae=float(mean_absolute_error(y_true, y_score)),
            mse=float(mean_squared_error(y_true, y_score)),
        )

        return EvaluationResult(
            y_true=np.asarray(y_true),
            y_score=np.asarray(y_score),
            y_pred=y_pred,
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            classification_threshold=float(threshold),
            metrics=metrics,
        )
