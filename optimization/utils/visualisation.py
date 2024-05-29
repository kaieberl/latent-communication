import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def pca_def(v):
    pca = PCA(n_components=2)
    pca.fit(v)
    return pca


def get_latent_space_data(model, train_loader, DEVICE, N=1000):
    images, _ = next(iter(train_loader))
    latent_spaces = []
    all_labels = []

    for images, labels in train_loader:
        #Exit for loop if all labels is as long as N
        if len(all_labels) > N:
            break
        images = images.to(DEVICE)
        #images = images.view(images.size(0), -1)
        latent_space = model.get_latent_space(images)
        latent_space = latent_space.cpu().detach().numpy()
        latent_spaces.append(latent_space)
        all_labels.append(labels.numpy())

    # Concatenate latent space representations from all batches
    latent_space = np.concatenate(latent_spaces, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return latent_space, all_labels


def plotLatentTransformed(latent_space, all_labels, A, pca1, name):
    # Transform the latent space
    latent_space_transformed = np.dot(latent_space, A.T)
    # Plot latent space via PCA
    latent_space_pca = pca1.transform(latent_space_transformed)
    # Plot the latent space
    plot = plt.scatter(latent_space_pca[:, 0], latent_space_pca[:, 1], c=all_labels, cmap='tab10', label=all_labels)
    plt.title('Latent Space '+ name)
    plt.show(plot)
    return plot


def avg_transformed_distances(z1, z2, A):
    # Compute the transformed distances
    distance_avg = 0
    for i in range(z1.shape[0]):
        distance_avg += np.linalg.norm(A @ z1[i] - z2[i])
    distance_avg /= z1.shape[0]
    return distance_avg
