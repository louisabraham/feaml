import torch
from torch.utils.data import TensorDataset
import sklearn.datasets
import numpy as np


def make_moons(
    n_samples=100, shuffle=True, noise=None, random_state=None, *, device=None
):
    X, y = sklearn.datasets.make_moons(n_samples, shuffle, noise, random_state)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    is_protected = torch.ByteTensor(X[:, 0] > np.median(X[:, 0]))
    if device is not None:
        X = X.to(device)
        y = y.to(device)
        is_protected = is_protected.to(device)
    return TensorDataset(X, y, is_protected)


def load_adult(*, valid_split=0.25, path="adult.npz", device=None):
    data = np.load(path)

    datasets = []
    for suffix in ("train", "test"):
        # the 13th column is always zero
        X = torch.FloatTensor(np.delete(data[f"x_{suffix}"], 13, 1))
        y = torch.FloatTensor(data[f"y_{suffix}"][:, 0])
        is_protected = torch.ByteTensor(np.cast[int](data[f"attr_{suffix}"][:, 0] > 0))
        X = X.to(device)
        y = y.to(device)
        is_protected = is_protected.to(device)
        datasets.append(TensorDataset(X, y, is_protected))
    train, test = datasets
    in_valid = torch.rand(len(train)) < valid_split
    train, valid = TensorDataset(*train[~in_valid]), TensorDataset(*train[in_valid])
    return train, valid, test


def abx_data_gen(dataset, x_protected: bool, batch_size: int, reject: int):

    X, _, is_protected = dataset.tensors
    X_pos = X[is_protected]
    X_neg = X[~is_protected]
    n_pos = len(X_pos)
    n_neg = len(X_neg)

    sample = lambda n: torch.randint(n, size=(batch_size, reject))

    while True:
        if x_protected:
            A = X_pos[sample(n_pos)]
            B = X_neg[sample(n_neg)]
            X = X_pos[torch.randint(n_pos, size=(batch_size,))]
        else:
            A = X_neg[sample(n_neg)]
            B = X_pos[sample(n_pos)]
            X = X_neg[torch.randint(n_neg, size=(batch_size,))]
        yield A, B, X

