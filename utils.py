import torch
from torch import nn


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


def sample_params():
    return dict(
        BATCH_SIZE=32,
        ABX_BATCH_SIZE=512,
        ABX_REJECT=1,
        A_abx=1,
        A_y=0,
        A_x=1,
        INPUT_SPACE=len(train[0][0]),
        LATENT_SPACE=16,
        HIDDEN_LAYERS=(32, 16),
        CLASSIFIER_HIDDEN_LAYERS=(32, 16),
        MAX_EPOCHS=100,
        MAX_EPOCHS_WITOUT_IMPROVEMENT=10,
        N_ROLLING_BATCHES=50,
    )

