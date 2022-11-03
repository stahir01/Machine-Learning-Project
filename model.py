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
        self.conv13 = nn.Conv2d(512, 256, 3)
        self.conv14 = nn.Conv2d(256, 256, 3)
        self.up_conv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv15 = nn.Conv2d(256, 128, 3)
        self.conv16 = nn.Conv2d(128, 128, 3)
        self.up_conv4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv17 = nn.Conv2d(128, 64, 3)
        self.conv18 = nn.Conv2d(64, 64, 3)
        self.conv19 = nn.Conv2d(64, 2, 1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        cnc_2_17 = self.crop(x, 568, 392)

        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        cnc_4_15 = self.crop(x, 280, 200)

        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        cnc_6_13 = self.crop(x, 136, 104)

        x = self.pool(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))

        cnc_8_11 = self.crop(x, 64, 56)

        x = self.pool(x)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.up_conv1(x)

        x = torch.cat((cnc_8_11,x), dim=1)
        x = F.relu(self.conv11(x))

        x = F.relu(self.conv12(x))
        x = self.up_conv2(x)

        x = torch.cat((cnc_6_13,x), dim=1)
        x = F.relu(self.conv13(x))

        x = F.relu(self.conv14(x))
        x = self.up_conv3(x)

        x = torch.cat((cnc_4_15,x), dim=1)
        x = F.relu(self.conv15(x))

        x = F.relu(self.conv16(x))
        x = self.up_conv4(x)

        x = torch.cat((cnc_2_17,x), dim=1)
        x = F.relu(self.conv17(x))
        
        x = F.relu(self.conv18(x))
        x = self.conv19(x)

        return x

    def crop(self, x, in_size, out_size):
        start = int((in_size - out_size)/2)
        stop  = int((in_size + out_size)/2)

        return x[:,:,start:stop,start:stop]

def build_model():
    
    model = UNet()

    return model

if __name__ == '__main__':
    model = build_model()

    X = torch.rand(1, 1, 572, 572)
    X = model(X)

    print(X.shape)