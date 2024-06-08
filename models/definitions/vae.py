from typing import List

import torch.nn as nn
import torch
from tqdm import tqdm
# get working directory
# import os
# Change directory
#os.chdir('/Users/mariotuci/Desktop/Google-Drive/Master/SoSe-24/Project Studies/Project/Code/latent-communication')
from models.definitions.base_model import BaseModel

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


class VAE(BaseModel):

    def __init__(self,in_dim: int,  dims: List[int], distribution_dim: int):
        super(VAE, self).__init__()

        self.in_dim = in_dim
        self.distribution_dim = distribution_dim
        self.dims = dims
        modules = []
        in_dim_layer = in_dim
        
        if dims is None:
            dims = [128, 256, 512]

        self.hidden_dim = dims[-1]

        for h_dim in dims:
            # Create Layers 
            modules.append(
                nn.Sequential(
                # Fully connected layer
                nn.Linear(in_dim_layer, h_dim),
                # Activation Function 
                nn.ReLU())
                )
                
            in_dim_layer = h_dim 
        
        # Variance and Mean for the latentspace distribution
        self.mu = nn.Linear(dims[-1], distribution_dim)
        self.var = nn.Linear(dims[-1], distribution_dim)


        # Build the encoder
        self.encoder = nn.Sequential(*modules)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(distribution_dim, dims[-1])

        for i in range(len(dims) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i - 1]),
                    nn.ReLU())
            )
  
        self.decoder = nn.Sequential(*modules)

        # Final Layer 
        self.final_layer = nn.Sequential(
            nn.Linear(dims[0], in_dim),
            nn.Tanh()
        )

    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the mean and log variance of the encoded input.
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        mu = self.mu(result)
        log_var = self.var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Takes the latent space representation and decodes it to the original input
        """

        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        """
        Passes through the encoder and decoder and returns the output.
        Using reparameterization trick to sample from the latent space.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
    
    def loss_function(self, x, x_reconst, mu, log_var):
        """
        Returns the loss function for the VAE.
        """
        reconst_loss = nn.functional.mse_loss(x_reconst, x, reduction='sum')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconst_loss + kl_div
    
    def sample(self, num_samples):
        """
        Samples from the latent space and returns the decoded output.
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.distribution_dim)
            z = z.to(device)
            z = self.decode(z)
        return z

    def get_latent_space(self, x):
        """
        Returns the latent space representation of the input. Last Layer of the Encoder before the mean and variance.
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        return result
    
