import torch.nn as nn
import torchvision.models as models


class ResNetModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.resnet50(pretrained=True)

        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):

        return self.model(x)