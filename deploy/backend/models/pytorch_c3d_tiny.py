import torch.nn as nn
import torchvision.transforms as transforms

# my_tranform =transforms.Compose([
#     # transforms.Resize((32,32,32)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5,0.5))
# ])


class C3dTiny(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一个3d卷积组
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(in_channels=1, kernel_size=3, padding = 1, out_channels=64),
            # 原网络结构没有，新增的
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2), stride = (1,2,2))
        )
        #
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(in_channels=64, kernel_size=3, padding = 1, out_channels=128),
            # 原网络结构没有，新增的
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        self.drop_out1 = nn.Dropout(0.2)
        #
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(in_channels = 128, kernel_size=3, padding = 1, out_channels=256),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, kernel_size=3, padding = 1, out_channels=256),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        self.drop_out2 = nn.Dropout(0.2)
        #
        self.conv_block4 = nn.Sequential(
            nn.Conv3d(in_channels = 256, kernel_size = 3, padding = 1, out_channels=512),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(in_channels = 512, kernel_size = 3, padding = 1, out_channels = 512),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        self.drop_out3 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        #计算输入特征数量：
        # 原始输入为32x32x32，经过pool1(1,2,2)后变为32x16x16
        # 经过pool2(2,2,2)后变为16x8x8
        # 经过pool3(2,2,2)后变为8x4x4
        # 经过pool4(2,2,2)后变为4x2x2
        # 因此最终特征图大小为4x2x2，通道数为512
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 4 * 2 * 2, 512),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.drop_out1(x)
        x = self.conv_block3(x)
        x = self.drop_out2(x)
        x = self.conv_block4(x)
        x = self.drop_out3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x