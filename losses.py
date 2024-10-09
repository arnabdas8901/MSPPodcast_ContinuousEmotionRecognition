import torch
import torch.nn as nn

class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def return_ccc(self, y_pred, y_true):
        mean_true = torch.mean(y_true)
        mean_pred = torch.mean(y_pred)
        centered_true = y_true - mean_true
        centered_pred = y_pred - mean_pred

        rho = 2.0 * torch.mean(centered_true * centered_pred) / \
              (torch.mean(centered_true ** 2) + torch.mean(centered_pred ** 2) + (mean_true - mean_pred) ** 2 + 1e-9)

        return rho

    def forward(self, y_pred, y_true):
        rho = self.return_ccc(y_pred, y_true)
        ccc_loss = 1.0 - rho

        return ccc_loss


