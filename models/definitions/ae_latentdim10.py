import torch
from models.definitions.base_model import LightningBaseModel

import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
"""
This uses the architecture of the relreps autoencoder with a smaller latent space dim
of 10 
"""

class VerySmallAutoencoder(LightningBaseModel):
    def __init__(self,hidden_dim=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.autoencoder = nn.Module()
        self.autoencoder.encoder = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),),
        nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()),
        nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()),
        nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )])
        self.autoencoder.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.GELU()),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU()),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU()),
            nn.Sequential(
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU()),
            nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()),
        ])
        self.autoencoder.encoder_out = nn.Sequential(
            nn.Linear(256, self.hidden_dim),
            nn.GELU(),
        )
        self.autoencoder.decoder_in = nn.Sequential(
            nn.Linear(self.hidden_dim, 1024),
            nn.GELU(),
        )
        

    def encode(self, x):
        for layer in self.autoencoder.encoder:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.autoencoder.encoder_out(x)
        return x

    def decode(self, x):
        x = self.autoencoder.decoder_in(x)
        x = x.view(x.size(0), 256, 2, 2)
        for layer in self.autoencoder.decoder:
            x = layer(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out

    def init_weights(self, m):
        if type(m) == nn.Linear:
            init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)
    

    def training_step(self, batch):
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch):
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)
        return loss

    
    def get_latent_space(self, x):
        for layer in self.autoencoder.encoder:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.autoencoder.encoder_out(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer