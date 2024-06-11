import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import torch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_difference_matrix(matrix, title=None, show_fig=True):
    """
    Plots a difference matrix as an image with a colormap.
    
    Args:
        matrix: 2D numpy array to be plotted.
        title: Title of the plot.
        show_fig: Boolean to indicate if the plot should be shown.
    """
    # Normalize the matrix values to range [0, 1]
    norm = mcolors.Normalize(vmin=np.min(matrix), vmax=np.max(matrix))
    matrix_normalized = norm(matrix)
    
    # Create a colormap: we use 'hot' which goes from black to red
    cmap = plt.cm.hot

    # Apply the colormap to the matrix data
    colored_matrix = cmap(matrix_normalized)

    # Plot the image
    plt.imshow(colored_matrix, interpolation='nearest')
    plt.colorbar(label='Normalized Value')
    if title:
        plt.title(title)
    plt.axis('off')  # Turn off the axis

    if show_fig:
        plt.show()

def visualize_image_error(image1, image2, show_fig=False):
    """
    Visualizes the error between two sets of images.

    Args:
        image1: First set of images.
        image2: Second set of images.
        show_fig: Boolean to indicate if the plot should be shown.

    Returns:
        errors: The error matrix between the images.
    """
    errors = np.abs(image1 - image2)
    if show_fig:
        plot_difference_matrix(errors, title='Image Error', show_fig=show_fig)
    return errors

def visualize_dataset_error(model1, model2, mapping, images):
    """
    Visualizes the error between two sets of images in a dataset.

    Args:
        model1: First model.
        model2: Second model.
        transformation: Transformation to be applied to latent space.
        images: A tensor or numpy array of shape (N, C, H, W), representing the images.

    Returns:
        None
    """
    errors = np.zeros_like(images[0])  # Initialize the error matrix with the shape of an individual image
    for i in range(len(images)):
        image = images[i]
        image = torch.tensor(image).unsqueeze(0)  # Add batch dimension if necessary
        
        with torch.no_grad():
            latents1 = model1.encode(image)
            recomposed = model2.decode(mapping.transform(latents1).float()).detach().numpy()

        errors += np.abs(image - recomposed)
    
    # Normalize errors
    errors /= len(images)
    plot_difference_matrix(errors, title='Dataset Error', show_fig=True)

# Ensure you have imported necessary libraries like torch and your model definitions.

