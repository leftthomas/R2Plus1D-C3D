import torch.nn as nn
import torch.nn.functional as F


channel_rate = 16


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=channel_rate,kernel_size=2,dilation=1)
        self.bn1=nn.BatchNorm2d(num_features=channel_rate)
        self.conv2 = nn.Conv2d(in_channels=channel_rate, out_channels=2*channel_rate, kernel_size=3, dilation=2)
        self.bn2 = nn.BatchNorm2d(num_features=2*channel_rate)
        self.conv3 = nn.Conv2d(in_channels=2*channel_rate, out_channels=4*channel_rate, kernel_size=3, dilation=3)
        self.bn3 = nn.BatchNorm2d(num_features=4*channel_rate)
        self.conv4 = nn.Conv2d(in_channels=4 * channel_rate, out_channels=4 * channel_rate, kernel_size=4, dilation=3)
        self.bn4 = nn.BatchNorm2d(num_features=4 * channel_rate)
        self.conv5 = nn.Conv2d(in_channels=4*channel_rate, out_channels=2*channel_rate, kernel_size=3, dilation=3)
        self.bn5 = nn.BatchNorm2d(num_features=2*channel_rate)
        self.conv6 = nn.Conv2d(in_channels=2*channel_rate, out_channels=channel_rate, kernel_size=3, dilation=2)
        self.bn6 = nn.BatchNorm2d(num_features=channel_rate)
        self.conv7 = nn.Conv2d(in_channels=channel_rate, out_channels=3, kernel_size=2, dilation=1)
        self.bn7 = nn.BatchNorm2d(num_features=3)

    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=F.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.leaky_relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.leaky_relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.leaky_relu(x)
        return x


if __name__ == '__main__':
    net=Net()
    print(net)


