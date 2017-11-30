import torch
import torch.nn.functional as F
from torch import nn

from capsulelayer import CapsuleLinear, CapsuleConv2d

config = {
    'MNIST': [16, '16D', 32, '32D', 64, '64D'],
    'CIFAR10': [16, '16D', 32, '32D', 64, '64D'],
    'CIFAR100': [16, '16D', 32, '32D', 64, '64D'],
    'STL10': [16, '16D', 32, '32D', 64, '64D'],
    'SVHN': [16, '16D', 32, '32D', 64, '64D'],
}


class SquashCapsuleNet(nn.Module):
    def __init__(self, in_channels, num_class, data_type):
        super(SquashCapsuleNet, self).__init__()
        self.features = self.make_layers(in_channels, config[data_type])
        self.classifier = CapsuleLinear(in_capsules=2 * 2 * 64 // 8, out_capsules=num_class, in_length=8,
                                        out_length=16)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), out.size(1), -1)
        out = torch.cat(out.chunk(num_chunks=out.size(1) // 8, dim=1), dim=2)
        out = out.transpose(1, 2)
        out = self.classifier(out)

        classes = (out ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)
        return classes

    @staticmethod
    def make_layers(in_channels, cfg):
        layers = []
        layers += [nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, stride=1),
                   nn.BatchNorm2d(16),
                   nn.ReLU(inplace=True)]
        in_channels = 16
        for x in cfg:
            if type(x) == str:
                x = int(x.replace('D', ''))
                layers += [CapsuleConv2d(in_channels, x, kernel_size=3, in_length=8, out_length=8, padding=1, stride=2)]
            else:
                layers += [CapsuleConv2d(in_channels, x, kernel_size=3, in_length=8, out_length=8, padding=1)]
            in_channels = x
        layers += [nn.AdaptiveAvgPool2d(2)]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    d = SquashCapsuleNet(in_channels=1, num_class=10, data_type='MNIST')
    print(d)
