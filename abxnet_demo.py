#%%
# %load_ext autoreload
# %autoreload 2
# %load_ext tensorboard.notebook
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm.autonotebook import tqdm

from datasets import *
from plots import *

BATCH_SIZE = 16

ABX_BATCH_SIZE = 512
ABX_REJECT = 5


dataset = make_moons(n_samples=10000, noise=0.1)

train_size = int(len(dataset) * 0.6)
val_size = int(len(dataset) * 0.1)
test_size = int(len(dataset) * 0.3)


plot_dataset(dataset)
#%%

from abxnet import ABXNet
from mlp import MLP

abxnet = ABXNet(
    encoder=MLP(3, 20, 20, 2),
    decoder=MLP(3, 20, 20, 2),
    classifier=None,  # MLP(2, 5, 1),
    A_abx=1,
    A_y=0,
    A_x=0.05,
    abx_scale=1,
)

#%%

writer = SummaryWriter(comment="test")
optimizer = optim.Adam(abxnet.parameters(), lr=1e-3)


for epoch in range(10000):

    running_losses = defaultfi
    for (batch, abx_pos, abx_neg) in zip(
        DataLoader(dataset, shuffle=True),
        abx_data_gen(
            dataset, x_protected=True, batch_size=ABX_BATCH_SIZE, reject=ABX_REJECT
        ),
        abx_data_gen(
            dataset, x_protected=False, batch_size=ABX_BATCH_SIZE, reject=ABX_REJECT
        ),
    ):

        optimizer.zero_grad()
        batch_loss, l_x, l_y = abxnet.loss(*batch)
        abx_loss = torch.max(
            abxnet.abx_loss(*abx_pos, x_protected=True),
            abxnet.abx_loss(*abx_neg, x_protected=False),
        )
        loss = batch_loss + abx_loss
        loss.backward()
        optimizer.step()

    writer.add_scalar("loss", loss.item())
    writer.add_scalar("abx_loss", abx_loss.item())
    writer.add_scalar("reconstruction loss", l_x.item())
    writer.add_scalar("classification loss", l_y.item())
    writer.close()

    if (epoch + 1) % 500 == 0:
        fig = plot_encoded(abxnet, dataset)
        writer.add_figure("encoded", fig)

with torch.no_grad():
    encoded, decoded, classif_pred = abxnet(X_torch, z_torch)

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
