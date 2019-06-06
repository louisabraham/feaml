#%%
# %load_ext autoreload
# %autoreload 2
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch import optim


#%%

X, y = sklearn.datasets.make_moons(1000, noise=0.1)
y = np.cast[bool](y)
z = X[:, 0] > np.median(X[:, 0])

X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y.astype("float"))
z_torch = torch.FloatTensor(z.astype("float"))
# plt.scatter(*X.T, c=y, marker=".")

#%%
plt.scatter(*X[y].T, c=z[y], marker="*")
plt.scatter(*X[~y].T, c=z[~y], marker="+")
plt.show()

#%%

from laftr import LAFTR, MLP

laftr = LAFTR(
    encoder=MLP(3, 5, 2),
    decoder=MLP(3, 5, 2),
    adversary=MLP(2, 5, 1),
    classifier=MLP(2, 5, 1),
)
#%%
fair_optimizer = optim.Adam(laftr.fair_parameters(), lr=1e-3)
adv_optimizer = optim.Adam(laftr.adv_parameters(), lr=1e-3)
for iter in range(5000):
    for optimizer, sign, steps in [(fair_optimizer, 1, 5), (adv_optimizer, -1, 1)]:
        for _ in range(steps):
            optimizer.zero_grad()
            loss = sign * laftr.loss(X_torch, z_torch, y_torch)
            loss.backward()
            optimizer.step()
    if (iter + 1) % 100 == 0:
        print(abs(loss.item()))


#%%
with torch.no_grad():
    encoded, decoded, classif_pred, adv_pred = laftr(X_torch, z_torch)

#%%
plt.title("dataset and decoded")
plt.scatter(*X.T, c="c", marker=".")
plt.scatter(*decoded.numpy().T, c="r", marker=".")

#%%
plt.title("encoded with sensitive")
# plt.scatter(*X.T, c="c", marker=".")
plt.scatter(*encoded.numpy().T, c=np.where(z, "w", "g"), marker=".")

#%%
plt.title("encoded with target")
plt.scatter(*encoded.numpy().T, c=np.where(y, "w", "g"), marker=".")

#%%
plt.title("encoded with predicted sensitive")
plt.scatter(
    *encoded.numpy().T, c=np.where(adv_pred.numpy()[:, 0] > 0, "w", "g"), marker="."
)

#%%
plt.title("encoded with predicted decoded")
plt.scatter(
    *encoded.numpy().T, c=np.where(classif_pred.numpy()[:, 0] > 0, "w", "g"), marker="."
)

#%%

plt.title("sensitive attributes")
plt.scatter(*X.T, c=np.where(z, "b", "r"), marker=".")
plt.scatter(*decoded.numpy().T, c=np.where(z, "w", "g"), marker=".", alpha=0.1)

#%%

plt.title("target")
plt.scatter(*X.T, c=np.where(y, "b", "r"), marker=".")
plt.scatter(*decoded.numpy().T, c=np.where(y, "w", "g"), marker=".", alpha=0.1)


#%%
representation = PCA(n_components=2).fit_transform(mapping.numpy())

plt.title("PCA of probabilistic mapping with sensitive attribute")
plt.scatter(*representation.T, c=np.where(z, "w", "g"), marker=".", alpha=0.1)
plt.show()
#%%

plt.title("PCA of probabilistic mapping with target")
plt.scatter(*representation.T, c=np.where(y, "w", "g"), marker=".", alpha=0.1)
plt.show()


#%%
