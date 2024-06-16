import numpy as np
import torch
from pathlib import Path
import os

from utils.visualization import visualize_mapping_error, visualize_latent_space_pca
from utils.dataloaders.dataloader_mnist_single import DataLoaderMNIST
from utils.model import load_model, get_transformations

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
device='cpu'

def load_mapping(config):
    name = f'/{config["mapping"]}_{config["model_name1"]}_{config["model_seed1"]}_{config["model_name2"]}_{config["model_seed2"]}_{config["num_samples"]}'
    path = config["storage_path"] + name
    if config["mapping"] == 'Linear':
        from optimization.optimizer import LinearFitting
        mapping = LinearFitting.from_file(path)
    elif config["mapping"] == 'Affine':
        from optimization.optimizer import AffineFitting
        mapping = AffineFitting.from_file(path)
    elif config["mapping"]  == 'NeuralNetwork':
        from optimization.optimizer import NeuralNetworkFitting
        mapping = NeuralNetworkFitting.from_file(path)
    else:
        raise ValueError("Invalid experiment name")
    return mapping

def get_latent_space_from_dataloader(model, dataloader,max_nr_batches=10):
    print("Calculating latent_spaces")
    latents = []
    labels = []
    N=0
    for data in dataloader:
        N+=1
        if N<max_nr_batches:
            image, label = data
            image, label = image.to(device), label.to(device)
            with torch.no_grad():
                latent = model.get_latent_space(image)
            latents.append(latent)
            labels.append(label)
        else:
            #skip for loop
            break
    print("Finished calculating latent_spaces")
    return torch.cat(latents), torch.cat(labels)

def main():
    seed1 = 1
    seed2 = 2
    seed3 = 3
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config ={ "base_dir": base_dir,
            "model_name1": "resnet",
            "model_path1": base_dir+f"/models/checkpoints/ResNet/MNIST/model_seed{seed1}.pth",
            "model_seed1": seed1,
            "model_name2": "resnet",
            "model_path2": base_dir+f"/models/checkpoints/ResNet/MNIST/model_seed{seed2}.pth",
            "model_seed2": seed2,
            "model_name3": "resnet",
            "model_path3": base_dir+f"/models/checkpoints/ResNet/MNIST/model_seed{seed3}.pth",
            "model_seed3": seed3,
            "mapping": "Linear",
            "num_samples": "100",
            "storage_path": base_dir+"/results/transformations/ResNet-LinearTransform"
    }
    
    #Load test data set to plot
    data_loader_model = DataLoaderMNIST(128, get_transformations(config["model_name1"]), seed=10, base_path = Path(config["base_dir"]))
    test_loader = data_loader_model.get_test_loader()
    print("loaded data")

    #Load model
    model1 = load_model(config["model_name1"], model_path=config["model_path1"])
    model2 = load_model(config["model_name2"],config["model_path2"])
    model3 = load_model(config["model_name3"],config["model_path3"])
    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)
    model1.eval()
    model2.eval()
    model3.eval()

    #Load mapping
    #mapping = load_mapping(config)

    #Get the latent space of the test set
    latents1,labels = get_latent_space_from_dataloader(model1, test_loader)
    latents2,_ = get_latent_space_from_dataloader(model2, test_loader)
    latents3,_ = get_latent_space_from_dataloader(model3, test_loader)

    latents1 = latents1.detach().numpy()
    latents2 = latents2.detach().numpy()
    latents3 = latents3.detach().numpy()
    labels = labels.detach().numpy()

    #Get transformed latent space
    #latents1_trafo = mapping.transform(latents1).detach().numpy()

    #Plot the results
    print("Plotting")
    fig_dir = "outputs/"
    #print(f"Mean error for {config["mapping"]} mapping: {np.mean(np.linalg.norm(latents2 - latents1_trafo, axis=1)):.4f}")
    pca,_  = visualize_latent_space_pca(latents1, labels, fig_dir + f"figures/latent_space_pca_{config["model_name1"]}_{seed1}_test.png")
    visualize_latent_space_pca(latents2, labels, fig_dir + f"figures/latent_space_pca_{config["model_name2"]}_{seed2}_test.png",pca = pca)
    visualize_latent_space_pca(latents3, labels, fig_dir + f"figures/latent_space_pca_{config["model_name3"]}_{seed3}_test.png",pca = pca)
    #visualize_latent_space_pca(latents1_trafo, labels, f"../../figures/latent_space_pca_{config["mapping"]}_{config["model_name1"]}_seed{seed1}_{config["model_name2"]}_seed{seed2}_test.png", pca=pca)
    #errors = np.linalg.norm(latents2 - latents1_trafo, axis=1)
    #visualize_mapping_error(latents2_2d, errors, f"../../figures/mapping_error_{cfg.model1.name}_seed{cfg.model1.seed}_{cfg.model1.name}_seed{cfg.model2.seed}_test_linear_.png")

if __name__ == "__main__":
    main()