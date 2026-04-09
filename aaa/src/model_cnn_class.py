import torch
import torch.nn as nn


class CNNClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128*28*28,256),
            nn.ReLU(),

            nn.Linear(256,6)   # 6 AQI classes
        )


    def forward(self,x):

        x = self.features(x)

        x = self.classifier(x)

        return x