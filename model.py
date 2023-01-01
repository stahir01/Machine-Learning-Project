import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding='same')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        h = x.clone()

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        return x, h

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(UpBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding='same')

    def forward(self, x, h):
        x = self.up_conv(x)
        c = self.crop(h, h.shape[2], x.shape[2])
        x = torch.cat((c, x), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x

    def crop(self, x, in_size, out_size):
        start = int((in_size - out_size)/2)
        stop  = int((in_size + out_size)/2)

        return x[:,:,start:stop,start:stop]


class NewUNet(nn.Module):
    def __init__(self) -> None:
        super(NewUNet, self).__init__()
        self.down_block_1 = DownBlock(in_channels=1, out_channels=64)
        self.down_block_2 = DownBlock(in_channels=64, out_channels=128)
        self.down_block_3 = DownBlock(in_channels=128, out_channels=256)
        self.down_block_4 = DownBlock(in_channels=256, out_channels=512)

        self.middle_conv_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding='same')
        self.middle_conv_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding='same')

        self.up_block_1 = UpBlock(in_channels=1024, out_channels=512)
        self.up_block_2 = UpBlock(in_channels=512, out_channels=256)
        self.up_block_3 = UpBlock(in_channels=256, out_channels=128)
        self.up_block_4 = UpBlock(in_channels=128, out_channels=64)

        self.out_conv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def forward(self, x):
        x, h_1 = self.down_block_1(x)
        x, h_2 = self.down_block_2(x)
        x, h_3 = self.down_block_3(x)
        x, h_4 = self.down_block_4(x)

        x = self.middle_conv_1(x)
        x = self.middle_conv_2(x)

        x = self.up_block_1(x, h_4)
        x = self.up_block_2(x, h_3)
        x = self.up_block_3(x, h_2)
        x = self.up_block_4(x, h_1)

        x = self.out_conv(x)

        x = F.softmax(x, dim=1)

        return x

class UNet(nn.Module):

    def __init__(self) -> None:
        super(UNet, self).__init__()
        self.pool  = nn.MaxPool2d(2, stride=2)

        # Contracting path
        self.conv1 = nn.Conv2d(1, 64, 3, padding='same')
        self.conv2 = nn.Conv2d(64, 64, 3, padding='same')

        self.conv3 = nn.Conv2d(64, 128, 3, padding='same')
        self.conv4 = nn.Conv2d(128, 128, 3, padding='same')

        self.conv5 = nn.Conv2d(128, 256, 3, padding='same')
        self.conv6 = nn.Conv2d(256, 256, 3, padding='same')

        self.conv7 = nn.Conv2d(256, 512, 3, padding='same')
        self.conv8 = nn.Conv2d(512, 512, 3, padding='same')

        self.conv9 = nn.Conv2d(512, 1024, 3, padding='same')
        self.conv10 = nn.Conv2d(1024, 1024, 3, padding='same')

        # Expansive path
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, 2, 2) # concat with copy and crop
        self.conv11 = nn.Conv2d(1024, 512, 3, padding='same')
        self.conv12 = nn.Conv2d(512, 512, 3, padding='same')
        self.up_conv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv13 = nn.Conv2d(512, 256, 3, padding='same')
        self.conv14 = nn.Conv2d(256, 256, 3, padding='same')
        self.up_conv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv15 = nn.Conv2d(256, 128, 3, padding='same')
        self.conv16 = nn.Conv2d(128, 128, 3, padding='same')
        self.up_conv4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv17 = nn.Conv2d(128, 64, 3, padding='same')
        self.conv18 = nn.Conv2d(64, 64, 3, padding='same')
        self.conv19 = nn.Conv2d(64, 2, 1, padding='same')
        # self.sm = nn.Softmax2d()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print("conv1: ", x.shape)
        x = F.relu(self.conv2(x))
        # print("conv2: ", x.shape)

        # cnc_2_17 = self.crop(x, 568, 392)
        cnc_2_17 = x

        x = self.pool(x)
        x = F.relu(self.conv3(x))
        # print("conv3: ", x.shape)
        x = F.relu(self.conv4(x))
        # print("conv4: ", x.shape)

        # cnc_4_15 = self.crop(x, 280, 200)
        cnc_4_15 = x

        x = self.pool(x)
        x = F.relu(self.conv5(x))
        # print("conv5: ", x.shape)
        x = F.relu(self.conv6(x))
        # print("conv6: ", x.shape)

        # cnc_6_13 = self.crop(x, 136, 104)
        cnc_6_13 = x

        x = self.pool(x)
        x = F.relu(self.conv7(x))
        # print("conv7: ", x.shape)
        x = F.relu(self.conv8(x))
        # print("conv8: ", x.shape)

        # cnc_8_11 = self.crop(x, 64, 56)
        cnc_8_11 = x

        x = self.pool(x)
        x = F.relu(self.conv9(x))
        # print("conv9: ", x.shape)
        x = F.relu(self.conv10(x))
        # print("conv10: ", x.shape)
        x = self.up_conv1(x)

        cnc_8_11 = self.crop(cnc_8_11, cnc_8_11.shape[2], x.shape[2])
        x = torch.cat((cnc_8_11,x), dim=1)
        x = F.relu(self.conv11(x))
        # print("conv11: ", x.shape)

        x = F.relu(self.conv12(x))
        x = self.up_conv2(x)
        # print("conv12: ", x.shape)

        cnc_6_13 = self.crop(cnc_6_13, cnc_6_13.shape[2], x.shape[2])
        x = torch.cat((cnc_6_13,x), dim=1)
        x = F.relu(self.conv13(x))
        # print("conv13: ", x.shape)

        x = F.relu(self.conv14(x))
        # print("conv14: ", x.shape)
        x = self.up_conv3(x)

        cnc_4_15 = self.crop(cnc_4_15, cnc_4_15.shape[2], x.shape[2])
        x = torch.cat((cnc_4_15,x), dim=1)
        x = F.relu(self.conv15(x))
        # print("conv15: ", x.shape)

        x = F.relu(self.conv16(x))
        # print("conv16: ", x.shape)
        x = self.up_conv4(x)

        cnc_2_17 = self.crop(cnc_2_17, cnc_2_17.shape[2], x.shape[2])
        x = torch.cat((cnc_2_17,x), dim=1)
        x = F.relu(self.conv17(x))
        # print("conv17: ", x.shape)
        
        x = F.relu(self.conv18(x))
        # print("conv18: ", x.shape)
        x = self.conv19(x)
        # print("conv19: ", x.shape)

        # x = self.sm(x)

        return x

    def crop(self, x, in_size, out_size):
        start = int((in_size - out_size)/2)
        stop  = int((in_size + out_size)/2)

        return x[:,:,start:stop,start:stop]

def build_model():
    
    model = UNet()

    return model

if __name__ == '__main__':
    # model = build_model()
    model = NewUNet()

    X = torch.rand(1, 1, 512, 512)
    X = model(X)

    print(X.shape)