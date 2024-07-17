"""Defines a linear mapping with dropout."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


class BayesianLinearRegression(LightningModule):
    def __init__(self, source_dim, target_dim, learning_rate, lamda=0) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.linear = nn.Linear(source_dim, target_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = F.dropout(self.linear(x), p=0.1, training=self.training)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        source_data, target_data = batch
        loss = F.mse_loss(self(source_data), target_data)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        source_data, target_data = batch
        loss = F.mse_loss(self(source_data), target_data)
        self.log('val_loss', loss)
        return loss
