from sklearn.ensemble import RandomForestClassifier


class RandomForestFactory:
    """Create RandomForest classifiers with one consistent configuration."""

    def __init__(
        self,
        *,
        n_estimators: int = 500,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def create(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
