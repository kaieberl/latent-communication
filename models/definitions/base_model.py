import torch.utils.data
from lightning import LightningModule
from abc import ABC, abstractmethod

import torch.nn as nn
from abc import abstractmethod

from tqdm import tqdm


class LightningBaseModel(LightningModule, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Takes the latent code as input and returns the reconstructed image.
        """
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input by passing through the encoder network
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes through the encoder and decoder and returns the output.
        Using reparameterization trick to sample from the latent space.
        """
        z = self.encode(x)
        return self.decode(z)

    @abstractmethod
    def get_latent_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the latent space representation of the input. Last Layer of the Encoder before the mean and variance.
        """
        pass

    @abstractmethod
    def configure_optimizers(self):
        """
        Define the optimizer for training.
        Override this method to specify your optimizer configuration.
        """
        pass


class BaseModel(nn.Module, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Takes the latent code as input and returns the reconstructed image.
        """
        pass

    @abstractmethod    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """"
        Encodes the input by passing through the encoder network
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes through the encoder and decoder and returns the output.
        Using reparameterization trick to sample from the latent space.
        """
        z = self.encode(x)
        return self.decode(z)

    @abstractmethod
    def get_latent_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the latent space representation of the input. Last Layer of the Encoder before the mean and variance.
        """
        pass

    @torch.no_grad()
    def get_latent_space_from_dataloader(self, dataloader):
        """
        Returns the latent space representation of the input.
        """
        total_samples = len(dataloader.dataset)
        latent_dim = self.hidden_dim
        latents = torch.zeros((total_samples, latent_dim), device=self.device)
        labels = torch.zeros(total_samples, dtype=torch.int, device=self.device)
        dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, shuffle=False)
        start_idx = 0
        for images, targets in tqdm(dataloader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            latents[start_idx:start_idx + dataloader.batch_size] = self.get_latent_space(images)
            labels[start_idx:start_idx + dataloader.batch_size] = targets
            start_idx += dataloader.batch_size
        return latents, labels