import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import torch.nn as nn
import torchvision
import torchvision.models as models

def visualize_latent_space_pca(latents,labels,fig_path=None,anchors=None):
    # Convert the 4D latent space to 2D using PCA
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents.view(latents.size(0), -1).cpu().detach().numpy())

    # Create a DataFrame for easy plotting
    latent_df = pd.DataFrame(latents_2d, columns=['PC1', 'PC2'])
    latent_df['Label'] = labels.numpy()

    # Plot the 2D latent space
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=latent_df, x='PC1', y='PC2', hue='Label', palette='tab10')
    plt.title('2D PCA of Latent Space')
    if anchors!=None:
        #plot anchors with star marker
        print(anchors.view(anchors.size(0), -1).cpu().detach().numpy().shape)
        anchors_2d = pca.transform(anchors.view(anchors.size(0), -1).cpu().detach().numpy())
        plt.scatter(anchors_2d[:,0],anchors_2d[:,1],marker='*',s=100,c='black')
    if fig_path !=None:
        plt.savefig(fig_path)
    plt.show()

def visualize_latent_space_tsne(latents,labels,perplexity=10,fig_path=None,anchors=None):
  
    latents_reshaped = latents.view(latents.shape[0], -1)  # Reshape to [20, 4*7*7]
    latents_np = latents_reshaped.cpu().detach().numpy()
    tsne = TSNE(n_components=2, random_state=1,perplexity=perplexity)
    latent_tsne = tsne.fit_transform(latents_np)

    # Create a DataFrame for seaborn plotting
    latent_df = pd.DataFrame(latent_tsne, columns=['Component 1', 'Component 2'])
    latent_df['Label'] = labels#.detach().numpy().astype(str)


    # Plot the 2D latent space
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=latent_df, x='Component 1', y='Component 2', hue='Label', palette='tab10', alpha=0.7)
    plt.title('t-SNE of Latent Space')
    plt.legend(title='Label', loc='upper right')
    if anchors!=None:
        #plot anchors with star marker
        print(anchors.view(anchors.size(0), -1).cpu().detach().numpy().shape)
        anchors_2d = pca.transform(anchors.view(anchors.size(0), -1).cpu().detach().numpy())
        plt.scatter(anchors_2d[:,0],anchors_2d[:,1],marker='*',s=100,c='black')
    plt.grid(True)
    if fig_path !=None:
        plt.savefig(fig_path)
    plt.show()

#Get latent vectors
def get_latent_vectors(encoder, data_loader, max_batches=None):
    encoder.eval()
    encoded_images = []
    labels_list = []
    num_batches_processed = 0

    for images, labels in data_loader:
        # Encode images using the encoder part of the model
        with torch.no_grad():  # Ensure no gradients are calculated
            encoded_output = encoder(images)

        # Append the encoded images and corresponding labels to lists
        encoded_images.append(encoded_output.cpu())  # Store on CPU
        labels_list.append(labels)

        num_batches_processed += 1
        if max_batches is not None and num_batches_processed >= max_batches:
            break  # Stop processing after reaching the specified max_batches

    # Concatenate the lists to create tensors for encoded images and labels
    encoded_images = torch.cat(encoded_images, dim=0)
    labels_tensor = torch.cat(labels_list, dim=0)

    return encoded_images, labels_tensor


