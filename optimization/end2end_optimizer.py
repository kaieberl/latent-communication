from pathlib import Path

import numpy as np
import cvxpy as cp
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch import loggers as pl_loggers

from optimization import optimizer
from optimization.mlp import MLP, End2EndMLP


class NeuralNetworkFitting:
    def __init__(self, train_loader, model1, model2, hidden_dim, lamda, val_loader=None, learning_rate=0.01, epochs=100):
        """
        Defines a neural network mapping between two latent spaces using an MLP.
        """
        self.epochs = epochs
        self.model = End2EndMLP(model1, model2, hidden_dim, lamda, learning_rate, epochs, train_loader, val_loader)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        """Encodes the image using model1, maps it to the target latent space using the MLP, and decodes it using
        model2.
        """
        return self.model(x)

    def define_loss(self):
        """
        Defines the loss function for the neural network fitting problem.
        """
        pass

    def fit(self):
        """
        Trains the neural network model using early stopping.
        """
        tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
        trainer = Trainer(max_epochs=self.epochs, min_epochs=self.epochs // 2, logger=tb_logger, log_every_n_steps=1,
                          callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)])
        val_loader = None
        if self.val_loader is not None:
            val_loader = DataLoader(TensorDataset(self.z1_val, self.z2_val), batch_size=self.z1_val.size(0),
                                    shuffle=False)
        trainer.fit(self.model, self.train_loader, val_loader)

    def get_results(self):
        """
        Returns the results of the optimization problem.

        Returns:
            torch.nn.Module: Trained model
        """
        return self.model.mapping

    def save_results(self, path):
        """
        Saves the model parameters to a file.

        Parameters:
            path (str): Path to save the model parameters
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.mapping, str(path) + '.pth')
        print("Model saved at ", path)

    def print_results(self):
        """
        Prints the results of the optimization problem.
        """
        print("Model: ", self.model.mapping)

    def transform(self, z1):
        """
        Applies the trained model to new data.

        Parameters:
            z1 (np.ndarray or torch.Tensor): New input data matrix of shape (n_samples, latent_dim1)

        Returns:
            torch.Tensor: Transformed data matrix of shape (n_samples, latent_dim2)
        """
        self.model.mapping.eval()
        if isinstance(z1, np.ndarray):
            z1 = torch.tensor(z1, dtype=torch.float32)
        with torch.no_grad():
            z1 = self.model.mapping(z1)
        return z1

    @staticmethod
    def from_file(path):
        """
        Returns an instance of class optimizer.NeuralNetworkFitting instead of the class itself

        Parameters:
            path (str): Path to the model parameters

        Returns:
            NeuralNetworkFitting: Instance of the class with the loaded model parameters
        """
        return optimizer.NeuralNetworkFitting.from_file(path)
