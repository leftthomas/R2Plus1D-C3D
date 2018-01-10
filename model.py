import torch.nn.functional as F
from torch import nn

from capsulelayer import CapsuleConv2d, CapsuleLinear


class MNISTCapsuleNet(nn.Module):
    def __init__(self):
        super(MNISTCapsuleNet, self).__init__()
        self.out_length = 16
        self.features = nn.Sequential(
            CapsuleConv2d(in_channels=1, out_channels=16, kernel_size=3, in_length=1, out_length=4, stride=1,
                          padding=1),
            CapsuleConv2d(in_channels=16, out_channels=16, kernel_size=3, in_length=4, out_length=4, stride=2,
                          padding=1),
            CapsuleConv2d(in_channels=16, out_channels=32, kernel_size=3, in_length=4, out_length=8, stride=1,
                          padding=1),
            CapsuleConv2d(in_channels=32, out_channels=32, kernel_size=3, in_length=8, out_length=8, stride=2,
                          padding=1),
            CapsuleConv2d(in_channels=32, out_channels=64, kernel_size=3, in_length=8, out_length=16, stride=1,
                          padding=1),
            CapsuleConv2d(in_channels=64, out_channels=64, kernel_size=3, in_length=16, out_length=16, stride=2,
                          padding=1)
        )
        self.classifier = CapsuleLinear(in_capsules=4 * 4 * 64 // 16, out_capsules=10, in_length=16,
                                        out_length=self.out_length)

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, self.out_length)

        out = self.classifier(out)
        classes = out.norm(p=2, dim=-1)
        classes = F.softmax(classes, dim=-1)
        return classes


class FashionMNISTCapsuleNet(nn.Module):
    def __init__(self):
        super(FashionMNISTCapsuleNet, self).__init__()
        self.out_length = 16
        self.features = nn.Sequential(
            CapsuleConv2d(in_channels=1, out_channels=32, kernel_size=5, in_length=1, out_length=8, stride=1,
                          padding=2),
            CapsuleConv2d(in_channels=32, out_channels=32, kernel_size=5, in_length=8, out_length=8, stride=2,
                          padding=2),
            CapsuleConv2d(in_channels=32, out_channels=64, kernel_size=3, in_length=8, out_length=16, stride=1,
                          padding=1),
            CapsuleConv2d(in_channels=64, out_channels=64, kernel_size=3, in_length=16, out_length=16, stride=2,
                          padding=1),
            CapsuleConv2d(in_channels=64, out_channels=128, kernel_size=3, in_length=16, out_length=16, stride=1,
                          padding=1),
            CapsuleConv2d(in_channels=128, out_channels=128, kernel_size=3, in_length=16, out_length=16, stride=2,
                          padding=1)
        )
        self.classifier = CapsuleLinear(in_capsules=4 * 4 * 128 // 16, out_capsules=10, in_length=16,
                                        out_length=self.out_length)

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, self.out_length)

        out = self.classifier(out)
        classes = out.norm(p=2, dim=-1)
        classes = F.softmax(classes, dim=-1)
        return classes


class CIFAR10CapsuleNet(nn.Module):
    def __init__(self):
        super(CIFAR10CapsuleNet, self).__init__()
        self.out_length = 16
        self.features = nn.Sequential(
            CapsuleConv2d(in_channels=3, out_channels=32, kernel_size=5, in_length=1, out_length=8, stride=1,
                          padding=2),
            CapsuleConv2d(in_channels=32, out_channels=32, kernel_size=5, in_length=8, out_length=8, stride=2,
                          padding=2),
            CapsuleConv2d(in_channels=32, out_channels=64, kernel_size=3, in_length=8, out_length=16, stride=1,
                          padding=1),
            CapsuleConv2d(in_channels=64, out_channels=64, kernel_size=3, in_length=16, out_length=16, stride=2,
                          padding=1),
            CapsuleConv2d(in_channels=64, out_channels=128, kernel_size=3, in_length=16, out_length=16, stride=1,
                          padding=1),
            CapsuleConv2d(in_channels=128, out_channels=128, kernel_size=3, in_length=16, out_length=16, stride=1,
                          padding=1),
            CapsuleConv2d(in_channels=128, out_channels=128, kernel_size=3, in_length=16, out_length=16, stride=2,
                          padding=1)
        )
        self.classifier = nn.Sequential(CapsuleLinear(in_capsules=4 * 4 * 128 // 16, out_capsules=32, in_length=16,
                                                      out_length=16),
                                        CapsuleLinear(in_capsules=32, out_capsules=10, in_length=16,
                                                      out_length=self.out_length))

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, self.out_length)

        out = self.classifier(out)
        classes = out.norm(p=2, dim=-1)
        classes = F.softmax(classes, dim=-1)
        return classes


class CIFAR100CapsuleNet(nn.Module):
    def __init__(self):
        super(CIFAR100CapsuleNet, self).__init__()
        self.out_length = 16
        self.features = nn.Sequential(
            CapsuleConv2d(in_channels=3, out_channels=32, kernel_size=3, in_length=1, out_length=8, stride=1,
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
                          padding=1)
        )
        self.classifier = CapsuleLinear(in_capsules=4 * 4 * 128 // 16, out_capsules=100, in_length=16,
                                        out_length=self.out_length)

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, self.out_length)

        out = self.classifier(out)
        classes = out.norm(p=2, dim=-1)
        classes = F.softmax(classes, dim=-1)
        return classes


class SVHNCapsuleNet(nn.Module):
    def __init__(self):
        super(SVHNCapsuleNet, self).__init__()
        self.out_length = 16
        self.features = nn.Sequential(
            CapsuleConv2d(in_channels=3, out_channels=32, kernel_size=3, in_length=1, out_length=8, stride=1,
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
                          padding=1)
        )
        self.classifier = CapsuleLinear(in_capsules=4 * 4 * 128 // 16, out_capsules=10, in_length=16,
                                        out_length=self.out_length)

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, self.out_length)

        out = self.classifier(out)
        classes = out.norm(p=2, dim=-1)
        classes = F.softmax(classes, dim=-1)
        return classes


class STL10CapsuleNet(nn.Module):
    def __init__(self):
        super(STL10CapsuleNet, self).__init__()
        self.out_length = 16
        self.features = nn.Sequential(
            CapsuleConv2d(in_channels=3, out_channels=32, kernel_size=3, in_length=1, out_length=8, stride=2,
                          padding=1),
            CapsuleConv2d(in_channels=32, out_channels=32, kernel_size=3, in_length=8, out_length=8, stride=2,
                          padding=1),
            CapsuleConv2d(in_channels=32, out_channels=64, kernel_size=3, in_length=8, out_length=16, stride=2,
                          padding=1),
            CapsuleConv2d(in_channels=64, out_channels=64, kernel_size=3, in_length=16, out_length=16, stride=2,
                          padding=1),
            CapsuleConv2d(in_channels=64, out_channels=128, kernel_size=3, in_length=16, out_length=16, stride=2,
                          padding=1),
            CapsuleConv2d(in_channels=128, out_channels=128, kernel_size=3, in_length=16, out_length=16, stride=2,
                          padding=1)
        )
        self.classifier = CapsuleLinear(in_capsules=2 * 2 * 128 // 16, out_capsules=10, in_length=16,
                                        out_length=self.out_length)

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, self.out_length)

        out = self.classifier(out)
        classes = out.norm(p=2, dim=-1)
        classes = F.softmax(classes, dim=-1)
        return classes


models = {'MNIST': MNISTCapsuleNet, 'FashionMNIST': FashionMNISTCapsuleNet, 'SVHN': SVHNCapsuleNet,
          'CIFAR10': CIFAR10CapsuleNet, 'CIFAR100': CIFAR100CapsuleNet, 'STL10': STL10CapsuleNet}
