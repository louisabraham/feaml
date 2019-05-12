#%%
%load_ext autoreload
%autoreload 2
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
z_torch = torch.ByteTensor(z.astype("uint8"))
# plt.scatter(*X.T, c=y, marker=".")

#%%
plt.scatter(*X[y].T, c=z[y], marker="*")
plt.scatter(*X[~y].T, c=z[~y], marker="+")
plt.show()

#%%

from lfr import LFR

lfr = LFR(features_dim=2, k=10)

optimizer = optim.Adam(lfr.parameters(), lr=5e-3)
for iter in range(10000):
    optimizer.zero_grad()
    loss = lfr.loss(X_torch, z_torch, y_torch)
    loss.backward()
    if (iter - 1) % 100 == 0:
        print(loss.item())
    optimizer.step()

#%%
with torch.no_grad():
    mapping, reconstructed, pred = lfr(X_torch, z_torch)

#%%
plt.title('dataset and prototypes')
plt.scatter(*X.T, c='c', marker='.')
plt.scatter(*lfr.centroids.detach().numpy().T, c='r', marker='X')
#%%

plt.title('sensitive attributes')
plt.scatter(*X.T, c=np.where(z, 'b', 'r'), marker=".")
plt.scatter(*reconstructed.numpy().T, c=np.where(z, 'w', 'g'), marker=".", alpha=.1)

#%%

plt.title('target')
plt.scatter(*X.T, c=np.where(y, 'b', 'r'), marker=".")
plt.scatter(*reconstructed.numpy().T, c=np.where(y, 'w', 'g'), marker=".", alpha=.1)

#%%
representation = PCA(n_components=2).fit_transform(mapping.numpy())

plt.title('PCA of probabilistic mapping with sensitive attribute')
plt.scatter(*representation.T, c=np.where(z, 'w', 'g'), marker=".", alpha=.1)
plt.show()
#%%

plt.title('PCA of probabilistic mapping with target')
plt.scatter(*representation.T, c=np.where(y, 'w', 'g'), marker=".", alpha=.1)
plt.show()



#%%
