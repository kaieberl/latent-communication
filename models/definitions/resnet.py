import torch.nn as nn
import torchvision.models as models
from models.definitions.base_model import BaseModel


class ResNet(BaseModel):
    def __init__(self, pretrained = True):
        super(ResNet, self).__init__()
        model = models.resnet18(pretrained=pretrained)
        self.hidden_dim = 512
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

        self.fc = nn.Linear(self.hidden_dim, 10)

    def encode(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def decode(self,z):
        out = self.fc(z)
        return out

    def get_latent_space(self, x):
        x = self.encode(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out