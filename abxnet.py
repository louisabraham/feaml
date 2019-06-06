import torch
from torch import nn
import torch.nn.functional as F

from utils import RMSELoss
from metrics import abx


class ABXNet(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        classifier=None,
        rec_loss=None,
        classif_loss=None,
        sigmoid_scale=1,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        if rec_loss is None:
            rec_loss = nn.MSELoss()
        self.rec_loss = rec_loss
        if classif_loss is None:
            classif_loss = nn.BCEWithLogitsLoss()
        self.classif_loss = classif_loss
        self.sigmoid_scale = sigmoid_scale

    def forward(self, x, is_protected):
        if len(is_protected.shape) == 1:
            is_protected = is_protected[:, None]
        is_protected = is_protected.type(torch.float)

        encoded = self.encoder(torch.cat((x, is_protected), dim=1))
        decoded = self.decoder(torch.cat((encoded, is_protected), dim=1))
        if self.classifier is not None:
            classif_pred = self.classifier(encoded)
        else:
            classif_pred = None

        return encoded, decoded, classif_pred

    def batch_loss(self, x, y, is_protected):
        if len(y.shape) == 1:
            y = y[:, None]

        encoded, decoded, classif_pred = self.forward(x, is_protected)

        # reconstruction error
        L_x = self.rec_loss(x, decoded)

        # prediction error
        L_y = (
            self.classif_loss(classif_pred, y)
            if classif_pred is not None
            else torch.tensor(0.0)
        )

        # total loss
        return L_x, L_y

    @staticmethod
    def euclidean_dist(a, b):
        return torch.sqrt(torch.sum((a - b) ** 2, dim=-1))

    def abx_loss(self, A, B, X, x_protected):

        if len(A.shape) == 2:
            A = A[:, None, :]
        if len(B.shape) == 2:
            B = B[:, None, :]
        X = X[:, None, :]

        def append(t, f):
            return torch.cat((t, f(t.shape[:-1] + (1,)).to(t.device)), dim=-1)

        if x_protected:
            A = append(A, torch.ones)
            B = append(B, torch.zeros)
            X = append(X, torch.ones)
        else:
            A = append(A, torch.zeros)
            B = append(B, torch.ones)
            X = append(X, torch.zeros)

        A, B, X = self.encoder(A), self.encoder(B), self.encoder(X)

        with torch.no_grad():
            AX = ((A - X) ** 2).sum(dim=-1)
            BX = ((B - X) ** 2).sum(dim=-1)

            mean_dist = (AX.mean(dim=1) + BX.mean(dim=1)) / 2

            # make the ones that are too close
            # very big
            AX[AX < 1e-12] = float("inf")

            AXmin, selectA = AX.min(dim=1)
            selectB = BX.argmin(dim=1)

            selected = ~torch.isinf(AXmin)
            selectA = selectA[selected]
            selectB = selectB[selected]

            mean_dist = mean_dist[selected]

        X = X[selected, 0]
        A = A[selected, selectA]
        B = B[selected, selectB]

        AX = ((A - X) ** 2).sum(dim=-1)
        BX = ((B - X) ** 2).sum(dim=-1)

        sub = BX - AX
        sub *= self.sigmoid_scale / mean_dist

        l_abx = torch.mean(torch.sigmoid(sub)) - 0.5

        return l_abx

    def valid_abx_loss(self, x, y, is_protected, n_sample=10 ** 6):
        with torch.no_grad():
            encoded, _, _ = self.forward(x, is_protected)
        encoded = encoded.numpy()
        is_protected = is_protected.numpy()
        return max(
            abx(encoded, is_protected, n_sample=n_sample),
            abx(encoded, ~is_protected, n_sample=n_sample),
        )
