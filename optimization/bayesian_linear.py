"""Defines a linear mapping with dropout."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


class BayesianLinearRegression(LightningModule):
    def __init__(self, source_dim, target_dim, learning_rate, lamda) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.A_aff = nn.Parameter(torch.randn(source_dim, target_dim))
        self.b_aff = nn.Parameter(torch.randn(target_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        # apply dropout to the weight elements of the model
        x = x @ self.A_aff + self.b_aff
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        source_data, target_data = batch
        # residuals = [self.A_aff @ source_data[i] + self.b_aff - target_data[i] for i in range(self.z1.shape[0])]
        # residuals = torch.stack(residuals)
        residuals = self(source_data) - target_data
        print(residuals.shape)
        loss = torch.norm(residuals, p=2, dim=1).mean()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        source_data, target_data = batch
        loss = F.mse_loss(self(source_data), target_data)
        self.log('val_loss', loss)
        return loss
