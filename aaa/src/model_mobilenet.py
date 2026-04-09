import torch.nn as nn
import torchvision.models as models


class MobileNetModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = models.mobilenet_v2(weights="DEFAULT")
        self.model.classifier[1] = nn.Linear(
            self.model.last_channel,
            1
        )

    def forward(self, x):
        return self.model(x)