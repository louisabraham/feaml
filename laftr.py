import torch
from torch import nn
import torch.nn.functional as F


from utils import RMSELoss


class LAFTR(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        adversary,
        classifier=None,
        rec_loss=None,
        adv_loss=None,
        classif_loss=None,
        A_x=1,
        A_y=1,
        A_z=50,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.adversary = adversary
        self.classifier = classifier
        if rec_loss is None:
            rec_loss = RMSELoss()
        self.rec_loss = rec_loss
        if adv_loss is None:
            adv_loss = nn.BCEWithLogitsLoss()
        self.adv_loss = adv_loss
        if classif_loss is None:
            classif_loss = nn.BCEWithLogitsLoss()
        self.classif_loss = classif_loss
        self.A_x = A_x
        self.A_y = A_y
        self.A_z = A_z

    def forward(self, x, is_protected):
        if len(is_protected.shape) == 1:
            is_protected = is_protected[:, None]

        encoded = self.encoder(torch.cat((x, is_protected), dim=1))
        decoded = self.decoder(torch.cat((encoded, is_protected), dim=1))
        if self.classifier is not None:
            classif_pred = self.classifier(encoded)
        else:
            classif_pred = None
        adv_pred = self.adversary(encoded)

        return encoded, decoded, classif_pred, adv_pred

    def loss(self, x, is_protected, y):
        if len(is_protected.shape) == 1:
            is_protected = is_protected[:, None]
        if len(y.shape) == 1:
            y = y[:, None]

        encoded, decoded, classif_pred, adv_pred = self.forward(x, is_protected)

        # reconstruction error
        L_x = self.rec_loss(x, decoded)

        # prediction error
        L_y = (
            self.classif_loss(classif_pred, y)
            if classif_pred is not None
            else torch.tensor(0.0)
        )

        # adversarial loss
        L_z = self.adv_loss(adv_pred, is_protected)

        # total loss
        return self.A_x * L_x + self.A_y * L_y - self.A_z * L_z

    def fair_parameters(self):
        for m in [self.encoder, self.decoder, self.classifier]:
            if m is not None:
                yield from m.parameters()

    def adv_parameters(self):
        yield from self.adversary.parameters()

