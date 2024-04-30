from latentis.project import RelativeProjector
from latentis.project import relative
from latentis.space import LatentSpace

import torch


with open('/Users/k/Documents/Code/latent-communication/0-shot-llm-vision/data/cifar10_vit_img.pt', 'rb') as f:
    emb1 = torch.load(f)

with open('/Users/k/Documents/Code/latent-communication/0-shot-llm-vision/data/cifar10_clip_img.pt', 'rb') as f:
    emb2 = torch.load(f)

indices = torch.randperm(emb1.size(0))[:1000]
space1 = LatentSpace(name="space1", vectors=emb1)
anchors1 = space1.vectors[indices]

space2 = LatentSpace(name="space2", vectors=emb2)
anchors2 = space2.vectors[indices]

projector1 = RelativeProjector(projection_fn=relative.cosine_proj)
projector2 = RelativeProjector(projection_fn=relative.cosine_proj)

rel_space1 = space1.to_relative(anchors=anchors1, projector=projector1)
rel_space2 = space2.to_relative(anchors=anchors2, projector=projector2)

with open('/Users/k/Documents/Code/latent-communication/0-shot-llm-vision/data/cifar10_vit_rel_img.pt', 'wb') as f:
    torch.save(rel_space1.vectors, f)

with open('/Users/k/Documents/Code/latent-communication/0-shot-llm-vision/data/cifar10_clip_rel_img.pt', 'wb') as f:
    torch.save(rel_space2.vectors, f)
