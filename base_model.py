import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    @abstractmethod
    def decode(self, z):
        """
        Takes the latent code as input and returns the reconstructed image.
        """
        pass

    @abstractmethod    
    def encode(self, x):
        """"
        Encodes the input by passing through the encoder network
        """
        pass

    @abstractmethod
    def forward(self, x):
        """
        Passes through the encoder and decoder and returns the output.
        Using reparameterization trick to sample from the latent space.
        """
        pass

    @abstractmethod
    def get_latent_space(self, x):
        """
        Returns the latent space representation of the input. Last Layer of the Encoder before the mean and variance.
        """
        pass