import cvxpy as cp
from abc import abstractmethod


class BaseOptimizer:
    def __init__(self, z1, z2):
        """
        Initializes the TransformOptimizer class.

        Parameters:
        z1 (np.ndarray): Input data matrix of shape (n_samples, latent_dim1)
        z2 (np.ndarray): Output data matrix of shape (n_samples, latent_dim2)
        """
        self.z1 = z1
        self.z2 = z2
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
        print("Defining the problem")
        loss = self.define_loss()
        objective = cp.Minimize(loss)
        self.problem = cp.Problem(objective)

    def fit(self):
        """
        Solves the optimization problem.
        """
        if self.problem is None:
            self.define_problem()
        print("Solving the problem")
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

    @abstractmethod
    def transform(self, z1):
        """
        Transforms the input data using the learned transformation.

        Parameters:
        z1 (np.ndarray): Input data matrix of shape (n_samples, latent_dim1)

        Returns:
        np.ndarray: Transformed data matrix of shape (n_samples, latent_dim2)
        """
        pass
