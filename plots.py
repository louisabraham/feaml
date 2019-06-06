import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_dataset(dataset, sample=1000):
    X, y, is_protected = map(
        torch.Tensor.numpy, dataset[torch.randperm(len(dataset))[:sample]]
    )
    y = np.cast[bool](y)
    plt.scatter(*X[y].T, c=is_protected[y], marker="*")
    plt.scatter(*X[~y].T, c=is_protected[~y], marker="+")
    return plt.gcf()


def plot_encoded(model, dataset, sample=1000):
    X, _, is_protected = dataset[torch.randperm(len(dataset))[:sample]]

    with torch.no_grad():
        encoded, decoded, classif_pred = model(X, is_protected)

    plt.title("encoded with sensitive")
    plt.scatter(
        *encoded.numpy().T,
        c=np.where(is_protected.numpy(), "k", "g"),
        marker=".",
        alpha=0.5
    )
    return plt.gcf()


def plot_decoded(model, dataset, sample=1000):
    X, _, is_protected = dataset[torch.randperm(len(dataset))[:sample]]

    with torch.no_grad():
        encoded, decoded, classif_pred = model(X, is_protected)

    plt.title("dataset and decoded")
    plt.scatter(*X.numpy().T, c="c", marker=".")
    plt.scatter(*decoded.numpy().T, c="r", marker=".")
    return plt.gcf()