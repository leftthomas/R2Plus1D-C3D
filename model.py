import torch
from torch import nn


def squash(tensor, dim=1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)


class SquashCapsuleNet(nn.Module):
    def __init__(self, in_channels, num_class):
        super(SquashCapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.lrelu = nn.LeakyReLU(0.2)

        self.adaavgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_class)

    def forward(self, x):
        batch_size = x.size(0)
        # capsules squash
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.lrelu(self.bn5(self.conv5(x)))
        x = torch.cat([squash(capsule) for capsule in torch.chunk(x, chunks=1, dim=1)], dim=1)

        x = self.adaavgpool(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    a = torch.FloatTensor([[0, 1, 2], [3, 4, 5]])
    b = squash(a)
    print(b)
    d = SquashCapsuleNet(in_channels=1, num_class=10)
    print(d)
