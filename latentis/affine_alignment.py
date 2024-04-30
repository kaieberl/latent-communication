import torch
from latentis import LatentSpace, transform
from latentis.estimate.affine import SGDAffineTranslator
from latentis.translate.translator import LatentTranslator
from latentis.estimate.dim_matcher import ZeroPadding


with open('/Users/k/Documents/Code/latent-communication/0-shot-llm-vision/data/cifar10_vit_img.pt', 'rb') as f:
    emb1 = torch.load(f)

with open('/Users/k/Documents/Code/latent-communication/0-shot-llm-vision/data/cifar10_clip_img.pt', 'rb') as f:
    emb2 = torch.load(f)

space1 = LatentSpace(vectors=emb1, name="space1")
space2 = LatentSpace(vectors=emb2, name="space2")

translator = LatentTranslator(
    random_seed=0,
    estimator=SGDAffineTranslator(),
    source_transforms=[transform.StandardScaling()],
    target_transforms=[transform.StandardScaling()],
)

translator.fit(source_data=space1, target_data=space2)

# write translated vectors to pt file
translated_vectors = translator(space1.vectors)
with open('/Users/k/Documents/Code/latent-communication/0-shot-llm-vision/data/cifar10_vit_img_to_clip_affine_img.pt', 'wb') as f:
    torch.save(translated_vectors, f)
