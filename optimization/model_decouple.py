import torch.nn as nn
import torch
from torch.optim import Adam
from lightning import LightningModule

"""
This file contains the definiton of an Affine model used for "Decoupling" approch
"""
class LinearModel(LightningModule):
    """
    defines the A1 and A2 where A1 maps z1 to z2 and A2 z2 to z1
    """
    def __init__(self, latent_dim1, latent_dim2, lamda, learning_rate=0.01):
        """
        latent_dim1: latent dimension of first model
        latent_dim2: latent dimesnion of second model
        lamda: regularization parameter used for decoupling
        """
        super().__init__()
        self.A1 = nn.Parameter(torch.randn(latent_dim2, latent_dim1))
        self.A2 = nn.Parameter(torch.randn(latent_dim1, latent_dim2))
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.loss_history = []

    def forward(self, z1, z2):
        """
        z1: shape (batch_size, latent_dim1)
        z2: shape (batch_size, latent_dim2)
        """
        residuals = torch.mean(torch.norm(self.A1 @ z1.T - z2.T, dim=0)**2 + torch.norm(self.A2 @ z2.T - z1.T, dim=0)**2)
        identity_reg = torch.mean(torch.norm(self.A2 @ (self.A1 @ z1.T) - z1.T, dim=0))
        loss = residuals + self.lamda * identity_reg
        return loss

    def training_step(self, batch, batch_idx):
        z1, z2 = batch
        loss = self.forward(z1, z2)
        self.loss_history.append(loss.item())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
    

class AffineModel(LightningModule):
    """
    defines the A1 and A2 where A1 maps z1 to z2 and A2 z2 to z1
    """
    def __init__(self, latent_dim1, latent_dim2, lamda, learning_rate=0.01):
        """
        latent_dim1: latent dimension of first model
        latent_dim2: latent dimesnion of second model
        lamda: regularization parameter used for decoupling
        """
        super().__init__()
        self.A1 = nn.Parameter(torch.randn(latent_dim2, latent_dim1))
        self.A2 = nn.Parameter(torch.randn(latent_dim1, latent_dim2))
        self.b1 = nn.Parameter(torch.randn(latent_dim1))
        self.b2 = nn.Parameter(torch.randn(latent_dim2))
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.loss_history = []

    def forward(self, z1, z2):
        """
        z1: shape (batch_size, latent_dim1)
        z2: shape (batch_size, latent_dim2)
        """
        residuals = torch.mean(torch.norm(self.A1 @ z1.T + self.b1 - z2.T, dim=0)**2 + torch.norm(self.A2 @ z2.T + self.b2 - z1.T, dim=0)**2)
        identity_reg = torch.mean(torch.norm(self.A2 @ (self.A1 @ z1.T + self.b1) + self.b2- z1.T, dim=0))
        loss = residuals + self.lamda * identity_reg
        return loss

    def training_step(self, batch, batch_idx):
        z1, z2 = batch
        loss = self.forward(z1, z2)
        self.loss_history.append(loss.item())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)