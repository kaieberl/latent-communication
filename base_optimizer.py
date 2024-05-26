# Import relevant libraries
import numpy as np
import cvxpy as cp
from abc import abstractmethod

class Base_Optimizer(self,z1,z2):
    def __init__(self, z1, z2, lamda):
        """
        Initializes the TransformOptimizer class.

        Parameters:
        z1 (np.ndarray): Input data matrix of shape (n_samples, latent_dim1)
        z2 (np.ndarray): Output data matrix of shape (n_samples, latent_dim2)
        lamda (float): Regularization parameter
        """
        self.z1 = z1
        self.z2 = z2
        self.lamda = lamda
        self.latent_dim1 = z1.shape[1]
        self.latent_dim2 = z2.shape[1]

    @abstractmethod
    def define_loss(self):
        """
        Defines the loss function for the optimization problem.
        """
        pass

    def define_problem(self):
        """
        Defines the optimization problem.
        """
        loss = self.define_loss()
        objective = cp.Minimize(loss)
        self.problem = cp.Problem(objective)

    def solve_problem(self):
        """
        Solves the optimization problem.
        """
        if self.problem is None:
            self.define_problem()
        self.problem.solve()

    @abstractmethod
    def get_results(self):
        """
        Returns the results of the optimization problem.
        """
        pass


    @abstractmethod    
    def save_results(self, path):
        """
        Saves the results of the optimization problem.

        Parameters:
        path (str): Path to save the results
        """
        pass

    @abstractmethod    
    def print_results(self):
        """
        Prints the results of the optimization problem.
        """
        pass