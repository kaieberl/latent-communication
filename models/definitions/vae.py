
import torch
import torch.nn as nn
from models.definitions.base_model import BaseModel

class Encoder(nn.Module):
    """Encoder network that encodes the input space into a latent space."""
    def __init__(self, in_dim, latent_dim):
        """Initialize an Encoder module.
        
        Args:
            in_dim: int, Number of input dimensions.
            latent_dim: int, Dimensionality of the latent space.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),  # corrected from 256
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        """Forward pass through the encoder.
        
        Args:
            x: torch.Tensor, Input tensor.
        Returns:
            torch.Tensor, Encoded representation.
        """
        return self.net(x)

class Distribution(nn.Module):
    """Distribution network that predicts the mean and variance of the latent space."""
    def __init__(self, latent_dim):
        """Initialize a Distribution module.
        
        Args:
            latent_dim: int, Dimensionality of the latent space.
        """
        super().__init__()
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        """Forward pass through the distribution network.
        
        Args:
            x: torch.Tensor, Input tensor.
        Returns:
            torch.Tensor, Mean of the latent space.
            torch.Tensor, Log variance of the latent space.
        """
        return self.mu(x), self.logvar(x)

class Decoder(nn.Module):
    """Decoder network that decodes the latent space into the input space."""
    def __init__(self, out_dim, latent_dim):
        """Initialize a Decoder module.
        
        Args:
            out_dim: int, Number of output dimensions.
            latent_dim: int, Dimensionality of the latent space.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        """Forward pass through the decoder.
        
        Args:
            x: torch.Tensor, Input tensor.
        Returns:
            torch.Tensor, Decoded representation.
        """
        return self.net(x)

class VAE(BaseModel):
    def __init__(self, in_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_dim, latent_dim)
        self.distribution = Distribution(latent_dim)
        self.decoder = Decoder(in_dim, latent_dim)
        self.return_var = False

    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the mean and log variance of the encoded input.
        """
        result = self.encoder(x)
        mu, log_var = self.distribution(result)
        return mu, log_var


  
    def decode(self, z):
        """
        Decodes the latent space representation
        """

        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Passes through the encoder and decoder and returns the output.
        Using reparameterization trick to sample from the latent space.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        if self.return_var:
            return x_reconst, mu, log_var
        return x_reconst

    def loss_function(self, x, x_reconst, mu, log_var):
        """
        Returns the loss function for the VAE.
        """
        reconst_loss = nn.functional.mse_loss(x_reconst, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconst_loss + kl_div

    def reconstruction_loss(self, x, x_reconst):
        """
        Returns the reconstruction loss of the VAE.
        """
        return nn.functional.mse_loss(x_reconst, x, reduction='sum')


    def get_latent_space(self, x):
        """
        Returns the latent space representation of the input.
        """
        mu, log_var = self.encode(x)
        return self.reparameterize(mu, log_var)

    