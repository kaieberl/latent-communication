import torch

dataset = 'cifar10'
model_name = 'vit'
with open(f'data/{dataset}_{model_name}_img.pt', 'rb') as f:
    image_representations_first_half = torch.load(f, map_location='cpu')

with open(f'data/{dataset}_{model_name}_img_2.pt', 'rb') as f:
    image_representations_second_half = torch.load(f, map_location='cpu')

image_representations = torch.cat((image_representations_first_half, image_representations_second_half), dim=0)
torch.save(image_representations, f'data/{dataset}_{model_name}_img.pt')
