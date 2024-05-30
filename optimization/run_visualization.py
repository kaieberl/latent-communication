import numpy as np
import torch

from vit.visualization import visualize_mapping_error, visualize_latent_space_pca

labels = torch.load(f"models/labels_test.pt", map_location='cpu')
latents1 = torch.load(f"models/latent_space_vit_seed1_test.pt", map_location='cpu').detach().numpy()
latents2 = torch.load(f"models/latent_space_vit_seed0_test.pt", map_location='cpu').detach().numpy()
A = np.load("../optimization/ViT-Linear/Linear_vit_0_vit_1_100.npy")
latents2_trafo = latents2 @ A.T

print(f"Mean error for linear mapping: {np.mean(np.linalg.norm(latents1 - latents2_trafo, axis=1)):.4f}")
pca, latents1_2d = visualize_latent_space_pca(latents1, labels, "figures/latent_space_pca_vit_seed1_test_.png")
errors = np.linalg.norm(latents1 - latents2_trafo, axis=1)
visualize_latent_space_pca(latents2_trafo, labels, "figures/latent_space_pca_vit_seed0_test_linear_.png", pca=pca)
visualize_mapping_error(latents1_2d, errors, "figures/mapping_error_vit_seed0_seed1_test_linear_.png")

data = np.load("../optimization/ViT-Linear/Affine_vit_0_vit_1_100.npz")
A, b = data['A'], data['b']
del data
latents2 = latents2 @ A.T
print(f"Mean error for affine mapping: {np.mean(np.linalg.norm(latents1 - latents2, axis=1)):.4f}")
errors = np.linalg.norm(latents1 - latents2, axis=1)
visualize_latent_space_pca(latents2, labels, "figures/latent_space_pca_vit_seed0_test_affine_.png", pca=pca)
visualize_mapping_error(latents1_2d, errors, "figures/mapping_error_vit_seed0_seed1_test_affine_.png")