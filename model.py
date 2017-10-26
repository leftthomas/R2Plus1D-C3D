import torch.nn as nn
import torch.nn.functional as F

channel_rate = 96


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=channel_rate, kernel_size=3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(num_features=channel_rate)

        self.conv2 = nn.Conv2d(in_channels=channel_rate, out_channels=2 * channel_rate, kernel_size=3, padding=2,
                               dilation=2)
        self.bn2 = nn.BatchNorm2d(num_features=2 * channel_rate)

        self.conv3 = nn.Conv2d(in_channels=2 * channel_rate, out_channels=3 * channel_rate, kernel_size=3, padding=3,
                               dilation=3)
        self.bn3 = nn.BatchNorm2d(num_features=3 * channel_rate)

        self.conv4 = nn.Conv2d(in_channels=3 * channel_rate, out_channels=3 * channel_rate, kernel_size=5, padding=6,
                               dilation=3)
        self.bn4 = nn.BatchNorm2d(num_features=3 * channel_rate)

        self.conv5 = nn.Conv2d(in_channels=3 * channel_rate, out_channels=2 * channel_rate, kernel_size=3, padding=3,
                               dilation=3)
        self.bn5 = nn.BatchNorm2d(num_features=2 * channel_rate)

        self.conv6 = nn.Conv2d(in_channels=2 * channel_rate, out_channels=channel_rate, kernel_size=3, padding=2,
                               dilation=2)
        self.bn6 = nn.BatchNorm2d(num_features=channel_rate)

        self.conv7 = nn.Conv2d(in_channels=channel_rate, out_channels=3, kernel_size=3, padding=1, dilation=1)
        self.bn7 = nn.BatchNorm2d(num_features=3)

        self.fc1 = nn.Linear(in_features=channel_rate * channel_rate * 3, out_features=48 * 48 * 2)
        self.fc2 = nn.Linear(in_features=48 * 48 * 2, out_features=24 * 24 * 1)
        self.fc3 = nn.Linear(in_features=24 * 24 * 1, out_features=10)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        # print(x.data.size())
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        # print(x.data.size())
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        # print(x.data.size())
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        # print(x.data.size())
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        # print(x.data.size())
        x = F.leaky_relu(self.bn6(self.conv6(x)))
        # print(x.data.size())
        x = F.leaky_relu(self.bn7(self.conv7(x)))
        # print(x.data.size())

        x = x.view(-1, channel_rate * channel_rate * 3)
        x = self.fc1(x)
        x = F.dropout(x)
        x = self.fc2(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    net = Net()
    print(net)
