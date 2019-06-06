import torch
from torch import nn
import torch.nn.functional as F


class LFR(nn.Module):
    def __init__(self, features_dim, k=5, A_x=1, A_y=1, A_z=50):
        super().__init__()
        self.features_dim = features_dim
        self.k = k
        self.A_x = A_x
        self.A_y = A_y
        self.A_z = A_z

        self.alpha_p = nn.Parameter(torch.empty(features_dim))
        self.alpha_np = nn.Parameter(torch.empty(features_dim))
        self.classif_w = nn.Parameter(torch.empty(k, 1))
        self.centroids = nn.Parameter(torch.empty(k, features_dim))

        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.alpha_p, a=0.9, b=1.1)
        nn.init.uniform_(self.alpha_np, a=0.9, b=1.1)
        nn.init.normal_(self.classif_w)
        nn.init.normal_(self.centroids)

    @staticmethod
    def M_nk(X, centroids, alpha):
        return F.softmax(LFR.distances(X, centroids, alpha), dim=-1)

    @staticmethod
    def distances(X, centroids, alpha):
        aux = (X[:, None, :] - centroids[None, :, :]) ** 2
        aux *= alpha[None, None, :]
        return aux.sum(-1)

    def forward(self, x, is_protected):
        n = len(x)
        mapping = torch.empty((n, self.k))
        mapping[is_protected] = self.M_nk(x[is_protected], self.centroids, self.alpha_p)
        mapping[~is_protected] = self.M_nk(
            x[~is_protected], self.centroids, self.alpha_p
        )

        # reconstructed values
        reconstructed = torch.mm(mapping, self.centroids)

        # predictions
        # apply the sigmoid to constraint the weights between 0 and 1
        pred = torch.mm(mapping, torch.sigmoid(self.classif_w)).flatten()

        return mapping, reconstructed, pred

    def loss(self, x, is_protected, y):
        mapping, reconstructed, pred = self.forward(x, is_protected)

        # reconstruction error
        # MSE instead of SE in standard paper
        L_x = F.mse_loss(x, reconstructed)

        # prediction error
        L_y = F.binary_cross_entropy(pred, y)

        # unfairness: absolute error between average mappings
        L_z = torch.sum(
            torch.abs(
                mapping[is_protected].mean(dim=0) - mapping[~is_protected].mean(dim=0)
            )
        )

        # total loss
        return self.A_x * L_x + self.A_y * L_y + self.A_z * L_z

