import torch
from torch import nn


def squash(tensor, dim=1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)


class SquashCapsuleNet(nn.Module):
    def __init__(self, in_channels, num_class):
        super(SquashCapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=7, stride=1, padding=3,
                               dilation=1,
                               groups=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=4, dilation=2,
                               groups=4)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=3, dilation=3,
                               groups=16)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=3, dilation=3,
                               groups=16)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=3, dilation=3,
                               groups=16)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=4, dilation=2,
                               groups=4)

        self.adaavgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1), nn.LeakyReLU(0.2),
                                        nn.Conv2d(256, num_class, kernel_size=1))

    def forward(self, x):
        batch_size = x.size(0)
        # capsules squash
        x = self.conv1(x)
        x = torch.cat([squash(capsule) for capsule in torch.chunk(x, chunks=1, dim=1)], dim=1)
        x = self.conv2(x)
        x = torch.cat([squash(capsule) for capsule in torch.chunk(x, chunks=4, dim=1)], dim=1)
        x = self.conv3(x)
        x = torch.cat([squash(capsule) for capsule in torch.chunk(x, chunks=16, dim=1)], dim=1)
        x = self.conv4(x)
        x = torch.cat([squash(capsule) for capsule in torch.chunk(x, chunks=16, dim=1)], dim=1)
        x = self.conv5(x)
        x = torch.cat([squash(capsule) for capsule in torch.chunk(x, chunks=16, dim=1)], dim=1)
        x = self.conv6(x)
        x = torch.cat([squash(capsule) for capsule in torch.chunk(x, chunks=4, dim=1)], dim=1)

        x = self.adaavgpool(x)
        x = self.classifier(x)
        x = x.view(batch_size, -1)
        return x


if __name__ == "__main__":
    a = torch.FloatTensor([[0, 1, 2], [3, 4, 5]])
    b = squash(a)
    print(b)
    d = SquashCapsuleNet(in_channels=1, num_class=10)
    print(d)
