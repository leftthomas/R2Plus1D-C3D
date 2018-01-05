import torch.nn.functional as F
from torch import nn

from capsulelayer import CapsuleConv2d, CapsuleLinear


class MNISTCapsuleNet(nn.Module):
    def __init__(self):
        super(MNISTCapsuleNet, self).__init__()
        self.features = nn.Sequential(
            CapsuleConv2d(in_channels=1, out_channels=32, kernel_size=3, in_length=1, out_length=8, stride=1,
                          padding=1),
            CapsuleConv2d(in_channels=32, out_channels=32, kernel_size=3, in_length=8, out_length=8, stride=2,
                          padding=1),
            CapsuleConv2d(in_channels=32, out_channels=64, kernel_size=3, in_length=8, out_length=16, stride=1,
                          padding=1),
            CapsuleConv2d(in_channels=64, out_channels=64, kernel_size=3, in_length=16, out_length=16, stride=2,
                          padding=1),
            CapsuleConv2d(in_channels=64, out_channels=128, kernel_size=3, in_length=16, out_length=16, stride=1,
                          padding=1),
            CapsuleConv2d(in_channels=128, out_channels=128, kernel_size=3, in_length=16, out_length=16, stride=2,
                          padding=1),
        )
        self.classifier = CapsuleLinear(in_capsules=4 * 4 * 128 // 16, out_capsules=10, in_length=16, out_length=16)

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, 16)

        out = self.classifier(out)
        classes = (out ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        return classes


class FashionMNISTCapsuleNet(nn.Module):
    def __init__(self):
        super(FashionMNISTCapsuleNet, self).__init__()
        self.features = nn.Sequential(
            CapsuleConv2d(in_channels=1, out_channels=64, kernel_size=3, in_length=1, out_length=8, stride=1,
                          padding=1),
            CapsuleConv2d(in_channels=64, out_channels=64, kernel_size=3, in_length=8, out_length=8, stride=2,
                          padding=1),
            CapsuleConv2d(in_channels=64, out_channels=128, kernel_size=3, in_length=8, out_length=16, stride=1,
                          padding=1),
            CapsuleConv2d(in_channels=128, out_channels=128, kernel_size=3, in_length=16, out_length=16, stride=2,
                          padding=1),
            CapsuleConv2d(in_channels=128, out_channels=256, kernel_size=3, in_length=16, out_length=16, stride=1,
                          padding=1),
            CapsuleConv2d(in_channels=256, out_channels=256, kernel_size=3, in_length=16, out_length=16, stride=2,
                          padding=1),
        )
        self.classifier = nn.Sequential(CapsuleLinear(in_capsules=4 * 4 * 256 // 16, out_capsules=128, in_length=16,
                                                      out_length=16),
                                        CapsuleLinear(in_capsules=128, out_capsules=10, in_length=16,
                                                      out_length=16))

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, 16)

        out = self.classifier(out)
        classes = (out ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        return classes


models = {
    'MNIST': MNISTCapsuleNet(),
    'FashionMNIST': FashionMNISTCapsuleNet(),
    'SVHN': ['32-4', '32-4D', '64-8', '64-8D', '128-16', '128-16', '128-16D', '256-16', '256-16', '256-16D'],
    'CIFAR10': ['32-4', '32-4D', '64-8', '64-8D', '128-16', '128-16', '128-16D', '256-16', '256-16', '256-16D'],
    'CIFAR100': ['32-4', '32-4D', '64-8', '64-8D', '128-16', '128-16', '128-16D', '256-16', '256-16', '256-16D'],
    'STL10': ['32-4', '32-4D', '64-8', '64-8D', '128-16', '128-16', '128-16D', '256-16', '256-16', '256-16D'],
}
