import torch.nn as nn
import torchvision
import torchvision.models as models
import sys
import os 
from base_model import BaseModel	


class ResNet(BaseModel):
    def __init__(self, pretrained = True):
        super(ResNet, self).__init__()
        model_resnet18 = models.resnet18(pretrained=pretrained)
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool

        self.fc = nn.Linear(512, 10)

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