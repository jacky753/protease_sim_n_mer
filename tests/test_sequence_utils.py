import numpy as np
import pandas as pd

from protease_rf.sequence_utils import labels_to_binary, normalize_sequence, sliding_windows


def test_normalize_sequence_preserves_length_with_x():
    assert normalize_sequence(" acd-e ") == "ACDXE"


def test_labels_to_binary_text():
    values = pd.Series(["positive", "neg", "+", "0"])
    result = labels_to_binary(values, "dummy.csv")
    assert np.array_equal(result, np.array([1, 0, 1, 0]))


def test_sliding_windows_preserves_original_off_by_one_behavior():
    seq = "ABCDE"
    assert sliding_windows(seq, 3, include_last=False) == ["ABC", "BCD"]
    assert sliding_windows(seq, 3, include_last=True) == ["ABC", "BCD", "CDE"]
