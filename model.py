import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool  = nn.MaxPool2d(2, stride=2)

        # Contracting path
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.conv7 = nn.Conv2d(256, 512, 3)
        self.conv8 = nn.Conv2d(512, 512, 3)
        self.conv9 = nn.Conv2d(512, 1024, 3)
        self.conv10 = nn.Conv2d(1024, 1024, 3)

        # Expansive path
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, 2, 2) # concat with copy and crop
        self.conv11 = nn.Conv2d(1024, 512, 3)
        self.conv12 = nn.Conv2d(512, 512, 3)
        self.up_conv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv13 = nn.Conv2d(1024, 512, 3)
        self.conv14 = nn.Conv2d(512, 512, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(x)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.up_conv1(x)

        return x

def build_model():
    
    model = UNet()

    return model

if __name__ == '__main__':
    model = build_model()

    X = torch.rand(1, 1, 572, 572)
    X = model(X)

    a = 1