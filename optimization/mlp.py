"""Defines a multi-layer perceptron used for mapping."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


class MLP(LightningModule):
    def __init__(self, source_dim, hidden_dim, target_dim, learning_rate, lamda) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.model = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.lamda)

    def training_step(self, batch, batch_idx):
        source_data, target_data = batch
        loss = F.mse_loss(self.model(source_data), target_data)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        source_data, target_data = batch
        loss = F.mse_loss(self.model(source_data), target_data)
        self.log('val_loss', loss)
        return loss
