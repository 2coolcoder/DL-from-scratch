"""Dataset loading, splitting and standardization.

scikit-learn supplies the raw arrays and the stratified split; everything the
model does with them afterwards is from scratch.
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split

__all__ = ["DATASETS", "Dataset", "load"]

DATASETS = {
    "iris": datasets.load_iris,
    "digits": datasets.load_digits,
    "wine": datasets.load_wine,
    "breast_cancer": datasets.load_breast_cancer,
}


class Dataset:
    def __init__(self, name, X_train, y_train, X_test, y_test, class_names):
        self.name = name
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.class_names = class_names
        self.n_features = len(X_train[0])
        self.n_classes = len(class_names)

    def __repr__(self):
        return (
            f"Dataset({self.name}: {len(self.X_train)} train / {len(self.X_test)} test, "
            f"{self.n_features} features, {self.n_classes} classes)"
        )


def _standardize(X_train, X_test):
    """Zero mean / unit variance, with statistics fit on the training split only."""
    n_feat = len(X_train[0])
    means, stds = [], []
    for j in range(n_feat):
        col = [row[j] for row in X_train]
        mu = sum(col) / len(col)
        var = sum((v - mu) ** 2 for v in col) / len(col)
        means.append(mu)
        stds.append(var ** 0.5 or 1.0)  # constant features stay at 0

    def apply(X):
        return [[(row[j] - means[j]) / stds[j] for j in range(n_feat)] for row in X]

    return apply(X_train), apply(X_test)


def load(name, test_size=0.25, seed=0, standardize=True):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    bunch = DATASETS[name]()
    X = bunch.data.tolist()
    y = [int(v) for v in bunch.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    if standardize:
        X_train, X_test = _standardize(X_train, X_test)
    names = [str(n) for n in getattr(bunch, "target_names", range(max(y) + 1))]
    return Dataset(name, X_train, y_train, X_test, y_test, names)
