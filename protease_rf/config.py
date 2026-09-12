from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


DEFAULT_PROTEASE_NUMBERS: Dict[str, int] = {
    "S01.247": 566,
    "S08.071": 741,
    "C01.032": 62,
    "S01.047": 503,
    "S01.021": 498,
    "S01.034": 502,
    "S01.087": 512,
    "S01.292": 578,
}


@dataclass(frozen=True)
class AppConfig:
    """Application configuration only.

    Keeping configuration in one place prevents the training, I/O and feature
    extraction classes from owning unrelated constants.
    """

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    target_protease: str = "S08.071"
    encoding_mode: str = "pbseq"
    sequence_length: int = 160
    cv_folds: int = 3
    split_dataset_dir_name: str = "S08071_withSpro_DefinitiveEdition"

    rf_n_estimators: int = 500
    random_state: int = 42
    rf_n_jobs: int = -1
    proteinbert_batch_size: int = 1

    # The original sliding-window function uses range(len(seq)-window), so the
    # final possible window is intentionally omitted. Keep this False to retain
    # identical behavior. Set True only if you explicitly want all windows.
    include_last_spike_window: bool = False

    protease_numbers: Dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_PROTEASE_NUMBERS)
    )

    @property
    def protease_turn(self) -> int:
        try:
            return self.protease_numbers[self.target_protease]
        except KeyError as exc:
            raise KeyError(
                f"Unknown protease {self.target_protease!r}. Add it to protease_numbers."
            ) from exc

    @property
    def split_dataset_dir(self) -> Path:
        return self.project_root / self.split_dataset_dir_name

    @property
    def output_dir(self) -> Path:
        return (
            self.project_root
            / "outputdir_rf_cls_priority"
            / self.encoding_mode
            / f"{self.protease_turn}_{self.target_protease}"
        )
