import torch
import torch.nn as nn
import torch.nn.functional as F
from models.definitions.base_model import LightningBaseModel
from models.definitions.resnet_block import ResBlock, PositionalNorm


class Encoder(nn.Module):
    """Encoder network that encodes the input space into a latent space."""

    def __init__(self, in_chan, latent_dim):
        """Initialize an Encoder module.
        
        Args:
            in_chan: int, Number of input channels of the images.
            latent_dim: int, Dimensionality of the latent space.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chan, 32, kernel_size=3, padding="same"),  # 32x32
            ResBlock(in_chan=32, out_chan=64, scale="downscale"),   # 16x16
            ResBlock(in_chan=64, out_chan=64),
            #ResBlock(in_chan=64, out_chan=64),
            #ResBlock(in_chan=64, out_chan=64),
            ResBlock(in_chan=64, out_chan=128, scale="downscale"),  # 8x8
            ResBlock(in_chan=128, out_chan=128),
            #ResBlock(in_chan=128, out_chan=128),
            #ResBlock(in_chan=128, out_chan=128),
            ResBlock(in_chan=128, out_chan=256, scale="downscale"), # 4x4
            ResBlock(in_chan=256, out_chan=256),
            #ResBlock(in_chan=256, out_chan=256),
            #ResBlock(in_chan=256, out_chan=256),
            PositionalNorm(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """Forward pass through the encoder.
        
        Args:
            x: torch.Tensor, Input tensor.
        Returns:
            torch.Tensor, Encoded representation.
        """
        return self.net(x)


class Decoder(nn.Module):
    """Decoder network that decodes the latent space back into the input space."""

    def __init__(self, out_chan, latent_dim):
        """Initialize a Decoder module.
        
        Args:
            out_chan: int, Number of output channels of the images.
            latent_dim: int, Dimensionality of the latent space.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.Unflatten(dim=-1, unflattened_size=(256, 4, 4)),     # 4x4
            ResBlock(in_chan=256, out_chan=128, scale="upscale"),   # 8x8
            ResBlock(in_chan=128, out_chan=128),
            #ResBlock(in_chan=128, out_chan=128),
            #ResBlock(in_chan=128, out_chan=128),
            ResBlock(in_chan=128, out_chan=64, scale="upscale"),    # 16x16
            ResBlock(in_chan=64, out_chan=64),
            #ResBlock(in_chan=64, out_chan=64),
            #ResBlock(in_chan=64, out_chan=64),
            ResBlock(in_chan=64, out_chan=32, scale="upscale"),     # 32x32
            ResBlock(in_chan=32, out_chan=32),
            #ResBlock(in_chan=32, out_chan=32),
            #ResBlock(in_chan=32, out_chan=32),
            PositionalNorm(32),
            nn.ReLU(),
            nn.Conv2d(32, out_chan, kernel_size=3, padding="same"),
        )

    def forward(self, x):
        """Forward pass through the decoder.
        
        Args:
            x: torch.Tensor, Input tensor.
        Returns:
            torch.Tensor, Decoded representation.
        """
        return self.net(x)


class Distribution(nn.Module):
    """Distribution network that maps the latent representation to mu and log_var."""

    def __init__(self, latent_dim):
        """Initialize a Distribution module.
        
        Args:
            latent_dim: int, Dimensionality of the latent space.
        """
        super().__init__()
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        """Forward pass to obtain mu and log_var.
        
        Args:
            x: torch.Tensor, Input tensor.
        Returns:
            tuple of torch.Tensor, (mu, log_var)
        """
        return self.mu(x), self.log_var(x)


class ResnetVAE(LightningBaseModel):
    """ResNet-based Variational Autoencoder."""

    def __init__(self, latent_dim, in_chan=3):
        """Initialize a ResnetVAE module.
        
        Args:
            latent_dim: int, Dimensionality of the latent space.
            in_chan: int, Number of input channels.
        """
        super(ResnetVAE, self).__init__()
        self.encoder = Encoder(in_chan, latent_dim)
        self.distribution = Distribution(latent_dim)
        self.decoder = Decoder(out_chan=in_chan, latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) from N(0, 1).
        
        Args:
            mu: torch.Tensor, Mean of the latent Gaussian.
            logvar: torch.Tensor, Log variance of the latent Gaussian.
        Returns:
            torch.Tensor, Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass through the VAE.
        
        Args:
            x: torch.Tensor, Input tensor.
        Returns:
            tuple of torch.Tensor, (reconstructed input, mu, logvar)
        """
        encoded = self.encoder(x)
        mu, logvar = self.distribution(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar

    def configure_optimizers(self):
        """Configure optimizers for training."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        """Training step.
        
        Args:
            batch: tuple of torch.Tensor, Batch of data.
            batch_idx: int, Batch index.
        Returns:
            torch.Tensor, Training loss.
        """
        x, _ = batch
        x_hat, mu, logvar = self(x)
        loss = self.loss_function(x_hat, x, mu, logvar)
        self.log('train_loss', loss)
        return loss

    def loss_function(self, recon_x, x, mu, logvar):
        """Compute the VAE loss function.
        
        Args:
            recon_x: torch.Tensor, Reconstructed input.
            x: torch.Tensor, Original input.
            mu: torch.Tensor, Mean of the latent Gaussian.
            logvar: torch.Tensor, Log variance of the latent Gaussian.
        Returns:
            torch.Tensor, VAE loss.
        """
        reconst_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconst_loss + kl_div

    def validation_step(self, batch, batch_idx):
        """Validation step.
        
        Args:
            batch: tuple of torch.Tensor, Batch of data.
            batch_idx: int, Batch index.
        Returns:
            torch.Tensor, Validation loss.
        """
        x, _ = batch
        x_hat, mu, logvar = self(x)
        loss = self.loss_function(x_hat, x, mu, logvar)
        self.log('val_loss', loss)
        return loss

    def encode(self, x):
        """Encode the input.
        
        Args:
            x: torch.Tensor, Input tensor.
        Returns:
            torch.Tensor, Encoded representation.
        """
        return self.encoder(x)

    def decode(self, x):
        """Decode the latent vector.
        
        Args:
            x: torch.Tensor, Latent vector.
        Returns:
            torch.Tensor, Decoded representation.
        """
        mu, log_var = self.distribution(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z)
    
    def get_latent_space(self, x):
        """Get the latent space representation.
        
        Args:
            x: torch.Tensor, Input tensor.
        Returns:
            torch.Tensor, Latent space representation.
        """
        return self.encoder(x)

def reparameterize(mu, logvar):
    """Reparameterization trick to sample from N(mu, var) from N(0, 1).
    
    Args:
        mu: torch.Tensor, Mean of the latent Gaussian.
        logvar: torch.Tensor, Log variance of the latent Gaussian.
    Returns:
        torch.Tensor, Sampled latent vector.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def vae_loss(recon_x, x, mu, logvar):
    """Compute the VAE loss function.
    
    Args:
        recon_x: torch.Tensor, Reconstructed input.
        x: torch.Tensor, Original input.
        mu: torch.Tensor, Mean of the latent Gaussian.
        logvar: torch.Tensor, Log variance of the latent Gaussian.
    Returns:
        torch.Tensor, VAE loss.
    """
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
