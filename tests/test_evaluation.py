import numpy as np

from protease_rf.evaluation import BinaryClassificationEvaluator


def test_evaluator_with_explicit_threshold():
    evaluator = BinaryClassificationEvaluator()
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.7, 0.9])

    result = evaluator.evaluate(
        y_true,
        y_score,
        classification_threshold=0.5,
    )

    assert result.classification_threshold == 0.5
    assert np.array_equal(result.y_pred, np.array([0, 0, 1, 1]))
    assert result.metrics.roc_auc == 1.0
    assert result.metrics.accuracy == 1.0
