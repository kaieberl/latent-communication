from pathlib import Path

import numpy as np
import cvxpy as cp
import torch
from lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from optimization.base_optimizer import BaseOptimizer
from optimization.mlp import MLP
from optimization.affinemodel import AffineModel


class AffineFitting(BaseOptimizer):
    def __init__(self, z1, z2, lamda, do_print=True):
        """
        Initializes the AffineFitting class.

        Parameters:
            z1 (np.ndarray): Input data matrix of shape (n_samples, latent_dim1)
            z2 (np.ndarray): Output data matrix of shape (n_samples, latent_dim2)
            lamda (float): Regularization parameter
        """
        super().__init__(z1, z2)
        self.lamda = lamda
        self.A_aff = cp.Variable((self.latent_dim2, self.latent_dim1))
        self.b_aff = cp.Variable(self.latent_dim2)
        self.problem = None
        self.do_print = do_print

    def define_loss(self):
        """
        Defines the loss function for the affine fitting problem.
        """
        n_samples = self.z1.shape[0]
        residuals = [self.A_aff @ self.z1[i] + self.b_aff - self.z2[i] for i in range(n_samples)]
        loss_aff = cp.norm2(cp.vstack(residuals))**2 + self.lamda * cp.norm(self.A_aff, 'fro')**2
        return loss_aff

    def get_results(self):
        """
        Returns the results of the optimization problem.

        Returns:
            float: Optimal value of the problem
            np.ndarray: Optimal values for A_aff
            np.ndarray: Optimal values for b_aff
        """
        print("Getting results")
        if self.problem is None:
            raise Exception("The problem has not been solved yet.")
        return self.problem.value, self.A_aff.value, self.b_aff.value

    def print_results(self):
        """
        Prints the results of the optimization problem.
        """
        opt_value, A_aff_opt, b_aff_opt = self.get_results()
        print("Optimal value: ", opt_value)
        print(A_aff_opt)
        print(b_aff_opt)

    def save_results(self, path):
        # As numpy arrays
        A = self.A_aff.value
        b = self.b_aff.value
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, A=A, b=b)
        if self.do_print:
            print("Results saved at ", path)
        return path

    def transform(self, z1):
        if isinstance(z1, torch.Tensor):
            z1 = z1.detach().cpu().numpy()
        return torch.from_numpy(z1 @ self.A_aff.value.T + self.b_aff.value)

    @classmethod
    def from_file(cls, path):
        """
        Loads the results of the optimization problem from a file.

        Parameters:
            path (str): Path to the file containing the results

        Returns:
            AffineFitting: Instance of the AffineFitting class with the loaded results
        """
        data = np.load(str(path) + '.npz')
        A = data['A']
        b = data['b']
        latent_dim2, latent_dim1 = A.shape
        instance = cls(np.zeros((1, latent_dim1)), np.zeros((1, latent_dim2)), 0)
        instance.A_aff = cp.Parameter((latent_dim2, latent_dim1), value=A)
        instance.b_aff = cp.Parameter(latent_dim2, value=b)
        return instance


class LinearFitting(BaseOptimizer):
    def __init__(self, z1, z2, lamda, do_print=True):
        """
        Initializes the LinearFitting class.

        Parameters:
            z1 (np.ndarray): Input data matrix of shape (n_samples, 32)
            z2 (np.ndarray): Output data matrix of shape (n_samples, 32)
            lamda (float): Regularization parameter
        """
        super().__init__(z1, z2)
        self.lamda = lamda
        self.A = cp.Variable((self.latent_dim2, self.latent_dim1))
        self.problem = None
        self.do_print = do_print

    def define_loss(self):
        """
        Defines the loss function for the matrix fitting problem.
        """
        n_samples = self.z1.shape[0]
        residuals = [self.A @ self.z1[i] - self.z2[i] for i in range(n_samples)]
        loss = cp.norm2(cp.vstack(residuals))**2 + self.lamda * cp.norm(self.A, 'fro')**2
        return loss

    def get_results(self):
        """
        Returns the results of the optimization problem.

        Returns:
            float: Optimal value of the problem
            np.ndarray: Optimal values for A
        """
        if self.problem is None:
            raise Exception("The problem has not been solved yet.")
        return self.problem.value, self.A.value

    def print_results(self):
        """
        Prints the results of the optimization problem.
        """
        opt_value, A_opt = self.get_results()
        print("Optimal value: ", opt_value)
        print(A_opt)

    def save_results(self, path):
        A = self.A.value
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, A)
        if self.do_print:
            print("Results saved at ", path)
        return path

    def transform(self, z1):
        if isinstance(z1, torch.Tensor):
            z1 = z1.detach().cpu().numpy()
        return torch.from_numpy(z1 @ self.A.value.T)

    @classmethod
    def from_file(cls, path):
        """
        Loads the results of the optimization problem from a file.

        Parameters:
            path (str): Path to the file containing the results

        Returns:
            LinearFitting: Instance of the LinearFitting class with the loaded results
        """
        A = np.load(str(path) + '.npy')
        latent_dim2, latent_dim1 = A.shape
        instance = cls(np.zeros((1, latent_dim1)), np.zeros((1, latent_dim2)), 0)
        instance.A = cp.Parameter((latent_dim2, latent_dim1), value=A)
        return instance


class NeuralNetworkFitting(BaseOptimizer):
    def __init__(self, z1, z2, hidden_dim, lamda, learning_rate=0.01, epochs=100, do_print=True):
        """
        Defines a neural network mapping between two latent spaces. Uses the MLP model.

        Parameters:
            z1 (np.ndarray): Input data matrix of shape (n_samples, latent_dim1)
            z2 (np.ndarray): Output data matrix of shape (n_samples, latent_dim2)
            hidden_dim (int): Number of neurons in the hidden layer
            lamda (float): Regularization parameter
            learning_rate (float): Learning rate for the optimizer
            epochs (int): Number of epochs to train the network
        """
        super().__init__(z1, z2)
        self.epochs = epochs
        self.model = MLP(self.latent_dim1, hidden_dim, self.latent_dim2, learning_rate, lamda)

        self.z1 = torch.tensor(z1, dtype=torch.float32)
        self.z2 = torch.tensor(z2, dtype=torch.float32)
        self.do_print = do_print

    def define_loss(self):
        """
        Defines the loss function for the neural network fitting problem.
        """
        pass

    def fit(self):
        """
        Trains the neural network model.
        """
        trainer = Trainer(max_epochs=self.epochs)
        trainer.fit(self.model, DataLoader(TensorDataset(self.z1, self.z2), batch_size=self.z1.size(0), shuffle=True))

    def get_results(self):
        """
        Returns the results of the optimization problem.

        Returns:
            torch.nn.Module: Trained model
        """
        return self.model

    def save_results(self, path):
        """
        Saves the model parameters to a file.

        Parameters:
            path (str): Path to save the model parameters
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model, str(path) + '.pth')
        print("Model saved at ", path)

    def print_results(self):
        """
        Prints the results of the optimization problem.
        """
        print("Model: ", self.model)

    def transform(self, z1):
        """
        Applies the trained model to new data.

        Parameters:
            z1 (np.ndarray or torch.Tensor): New input data matrix of shape (n_samples, latent_dim1)

        Returns:
            torch.Tensor: Transformed data matrix of shape (n_samples, latent_dim2)
        """
        self.model.eval()
        if isinstance(z1, np.ndarray):
            z1 = torch.tensor(z1, dtype=torch.float32)
        with torch.no_grad():
            z1 = self.model(z1)
        return z1

    @classmethod
    def from_file(cls, path):
        """
        Loads the results of the optimization problem from a file.

        Parameters:
            path (str): Path to the file containing the results

        Returns:
            NeuralNetworkFitting: Instance of the NeuralNetworkFitting class with the loaded results
        """
        model = torch.load(str(path) + '.pth')
        instance = cls(np.empty((1, 1)), np.empty((1, 1)), 0, 0)
        instance.model = model
        return instance


class KernelFitting(BaseOptimizer):
    def __init__(self, z1, z2, lamda, gamma, do_print=True):
        """
        Initializes the KernelFitting class with an RBF kernel.

        Parameters:
            z1 (np.ndarray): Input data matrix of shape (n_samples, n_features)
            z2 (np.ndarray): Output data matrix of shape (n_samples, n_outputs)
            lamda (float): Regularization parameter
            gamma (float): Parameter for the RBF kernel
        """
        super().__init__(z1, z2)
        self.z1 = z1
        self.z2 = z2
        self.lamda = lamda
        self.gamma = gamma
        self.n_samples = z1.shape[0]
        self.n_outputs = z2.shape[1]
        self.alpha = cp.Variable((self.n_samples, self.n_outputs))
        self.K = self.compute_kernel_matrix(z1)
        self.problem = None
        self.do_print = do_print

    def compute_kernel_matrix(self, z):
        """
        Computes the RBF kernel matrix.

        Parameters:
            z (np.ndarray): Data matrix

        Returns:
            np.ndarray: Kernel matrix
        """
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        sq_dists = np.sum(z**2, axis=1).reshape(-1, 1) + np.sum(z**2, axis=1) - 2 * np.dot(z, z.T)
        K = np.exp(-self.gamma * sq_dists)
        return K

    def define_loss(self):
        """
        Defines the loss function for the kernel fitting problem.
        """
        residuals = self.K @ self.alpha - self.z2
        loss = cp.norm(residuals, 'fro')**2 + self.lamda * cp.norm(self.alpha, 'fro')**2
        return loss

    def solve(self):
        """
        Solves the optimization problem.
        """
        loss = self.define_loss()
        self.problem = cp.Problem(cp.Minimize(loss))
        self.problem.solve()
        if self.do_print:
            print("Problem solved with status:", self.problem.status)

    def get_results(self):
        """
        Returns the results of the optimization problem.
        """
        if self.problem is None or self.problem.status != cp.OPTIMAL:
            raise Exception("The problem has not been solved yet or did not converge.")
        return self.problem.value, self.alpha.value

    def print_results(self):
        """
        Prints the results of the optimization problem.
        """
        opt_value, alpha_opt = self.get_results()
        print("Optimal value: ", opt_value)
        print("Optimal alpha coefficients:\n", alpha_opt)

    def transform(self, z1):
        """
        Transforms new input data using the learned model.

        Parameters:
            z1 (np.ndarray): New input data

        Returns:
            np.ndarray: Transformed data
        """
        K_new = self.compute_kernel_matrix(z1)
        return K_new @ self.alpha.value



class DecoupleFitting:
    def __init__(self, z1, z2, lamda, learning_rate=0.01, epochs=200, do_print=True):
        """
        Initializes the DecoupleFitting class.

        Parameters:
            z1 (np.ndarray): Input data matrix of shape (n_samples, latent_dim1)
           f z2 (np.ndarray): Output data matrix of shape (n_samples, latent_dim2)
            lamda (float): Regularization parameter
            learning_rate (float): Learning rate for the optimizer
            epochs (int): Number of epochs to train the model
            do_print (bool): Flag to print training progress
        """
        self.z1 = torch.tensor(z1, dtype=torch.float32)
        self.z2 = torch.tensor(z2, dtype=torch.float32)
        self.epochs = epochs
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.do_print = do_print

        self.mapping = AffineModel(self.z1.shape[1], self.z2.shape[1], lamda, learning_rate)


    def fit(self):
        """
        Trains the affine model.
        """
        trainer = Trainer(max_epochs=self.epochs)
        trainer.fit(self.mapping, DataLoader(TensorDataset(self.z1, self.z2), batch_size=self.z1.size(0), shuffle=True))

    def get_results(self):
        """
        Returns the results of the optimization problem.

        Returns:
            AffineModel: Trained model
        """
        return self.mapping.A1.detach().numpy(), self.mapping.A2.detach().numpy()

    def save_results(self, path):
        """
        Saves the model parameters to a file.

        Parameters:
            path (str): Path to save the model parameters
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.mapping.state_dict(), str(path) + '.pth')
        print("Model saved at ", path)

    def print_results(self):
        """
        Prints the results of the optimization problem.
        """
        print("A1:")
        print(self.mapping.A1.detach().numpy())
        print("A2:")
        print(self.mapping.A2.detach().numpy())

    def transform(self, z1):
        """
        Applies the trained model to new data.

        Parameters:
            z1 (np.ndarray or torch.Tensor): New input data matrix of shape (n_samples, latent_dim1)

        Returns:
            torch.Tensor: Transformed data matrix of shape (n_samples, latent_dim2)
        """
        self.mapping.eval()
        if isinstance(z1, np.ndarray):
            z1 = torch.tensor(z1, dtype=torch.float32)
        with torch.no_grad():
            z1_transformed = z1 @ self.mapping.A1.detach().T
        return z1_transformed.detach().numpy()

    @classmethod
    def from_file(cls, path):
        """
        Loads the results of the optimization problem from a file.

        Parameters:
            path (str): Path to the file containing the results

        Returns:
            DecoupleFitting: Instance of the DecoupleFitting class with the loaded results
        """
        # Create a dummy instance to get the latent dimensions
        dummy_instance = cls(np.empty((1, 1)), np.empty((1, 1)), 0, 0)
        latent_dim1 = dummy_instance.z1.shape[1]
        latent_dim2 = dummy_instance.z2.shape[1]

        # Create an instance with the correct dimensions and load the state_dict
        instance = cls(np.empty((1, latent_dim1)), np.empty((1, latent_dim2)), 0, 0)
        instance.model.load_state_dict(torch.load(str(path) + '.pth'))
        return instance
