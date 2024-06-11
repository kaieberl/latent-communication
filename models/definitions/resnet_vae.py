import torch
import torch.nn as nn
import torch.nn.functional as F

from models.definitions.base_model import LightningBaseModel
from models.definitions.resnet_ae import ResNetBlock


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.resblock = ResNetBlock(64)
        self.latent_layer = nn.Sequential(
            nn.Linear(64 * 7 * 7, latent_dim),
            nn.ReLU()
        )


    def forward(self, x):
        x = self.initial(x)
        x = self.resblock(x)
        x = x.view(x.size(0), -1)
        x = self.latent_layer(x)
        return x


class Distribution(nn.Module):
    def __init__(self, latent_dim):
        super(Distribution, self).__init__()
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.var = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.resblock = ResNetBlock(64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 64, 7, 7)
        z = self.resblock(z)
        z = self.final(z)
        return z


class ResnetVAE(LightningBaseModel):
    def __init__(self, latent_dim, in_channels):
        super(ResnetVAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.distribtuion = Distribution(latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

    def forward(self, x):
        x = self.encoder(x)
        mu, logvar = self.distribtuion(x)
        z = reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, logvar = self(x)
        loss = self.loss_function(x_hat, x, mu, logvar)
        self.log('train_loss', loss)
        return loss

    def loss_function(self, x, x_reconst, mu, log_var):
        reconst_loss = F.mse_loss(x_reconst, x, reduction='sum')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconst_loss + kl_div

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, logvar = self(x)
        loss = self.loss_function(x_hat, x, mu, logvar)
        self.log('val_loss', loss)
        return loss

    def encode(self, x):
        return self.encoder(x)

    def decode(self,x):
        mu, log_var = self.distribtuion(x)
        z = reparameterize(mu, log_var)
        return self.decoder(z)
    
    def get_latent_space(self, x):
        x = self.encoder(x)
        return x

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD