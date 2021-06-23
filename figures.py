import json

import matplotlib.pyplot as plt
import numpy as np

with open("results.json") as f:
    data = json.load(f)

del data[32]


def get(x):
    return np.array([d[x] for d in data])


def scale(x):
    return (x - x.min()) / (x.max() - x.min())


plt.xlabel("Accuracy")
plt.ylabel("Demographic parity")
plt.scatter(
    get("equalized odds Logistic"),
    get("demographic parity Logistic"),
    c="r",
    label="LR",
)
plt.scatter(
    get("equalized odds GBT"), get("demographic parity GBT"), c="b", label="GBT"
)
plt.legend()

plt.savefig("test.png")

plt.clf()


plt.xlabel("Accuracy")
plt.ylabel("Demographic parity")
plt.scatter(
    get("accuracy Logistic"), get("demographic parity Logistic"), c="r", label="LR"
)
plt.scatter(get("accuracy GBT"), get("demographic parity GBT"), c="b", label="GBT")
plt.legend()

plt.savefig("fig1.png")

plt.clf()


plt.xlabel("Accuracy")
plt.ylabel("Equalized odds")
plt.scatter(get("accuracy Logistic"), get("equalized odds Logistic"), c="r", label="LR")
plt.scatter(get("accuracy GBT"), get("equalized odds GBT"), c="b", label="GBT")
plt.legend()

plt.savefig("fig2.png")


plt.clf()

fig, ax = plt.subplots()
plt.xlabel("A_x")
plt.ylabel("A_y")
scatter = ax.scatter(
    get("A_x"),
    get("A_y"),
    # c=get("LATENT_SPACE"),
    s=20 + 500 * scale(get("accuracy Logistic")),
    facecolors="none",
    edgecolors="r",
    label="accuracy for LR",
)
plt.xlim(1e-3, 1e3)
plt.ylim(1e-3, 1e3)
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("fig3.png")


plt.clf()
fig, ax = plt.subplots()
plt.xlabel("A_x")
plt.ylabel("A_y")
scatter = ax.scatter(
    get("A_x"),
    get("A_y"),
    # c=get("LATENT_SPACE"),
    s=20 + 500 * scale(get("accuracy GBT")),
    facecolors="none",
    edgecolors="r",
    label="accuracy for GBT",
)
plt.xlim(1e-3, 1e3)
plt.ylim(1e-3, 1e3)
plt.xscale("log")
plt.yscale("log")
plt.legend()

plt.savefig("fig4.png")

plt.clf()


fig, ax = plt.subplots()
plt.xlabel("A_x")
plt.ylabel("A_y")
scatter = ax.scatter(
    get("A_x"),
    get("A_y"),
    # c=get("LATENT_SPACE"),
    s=20 + 500 * scale(get("equalized odds Logistic")),
    facecolors="none",
    edgecolors="r",
    label="equalized odds for LR",
)
plt.xlim(1e-3, 1e3)
plt.ylim(1e-3, 1e3)
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("fig5.png")


plt.clf()


fig, ax = plt.subplots()
plt.xlabel("A_x")
plt.ylabel("A_y")
scatter = ax.scatter(
    get("A_x"),
    get("A_y"),
    # c=get("LATENT_SPACE"),
    s=20 + 500 * scale(get("equalized odds GBT")),
    facecolors="none",
    edgecolors="r",
    label="equalized odds for GBT",
)
plt.xlim(1e-3, 1e3)
plt.ylim(1e-3, 1e3)
plt.xscale("log")
plt.yscale("log")
plt.legend()

plt.savefig("fig6.png")
