import torch.nn.functional as F
from torch import nn

from capsulelayer import CapsuleLinear, CapsuleConv2d

config = {
    'MNIST': [32, '32D', 64, '64D', 128, '128D', 256, 256],
    'CIFAR10': [32, '32D', 64, '64D', 128, '128D', 256, 256],
    'CIFAR100': [32, '32D', 64, '64D', 128, '128D', 256, 256],
    'STL10': [32, '32D', 64, '64D', 128, '128D', 256, 256],
    'SVHN': [32, '32D', 64, '64D', 128, '128D', 256, 256],
}


class SquashCapsuleNet(nn.Module):
    def __init__(self, in_channels, num_class, data_type):
        super(SquashCapsuleNet, self).__init__()
        self.features = self.make_layers(in_channels, config[data_type])
        self.classifier = CapsuleLinear(in_capsules=4 * 4 * 512 // 8, out_capsules=num_class, in_length=8,
                                        out_length=16)

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)

        classes = (out ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)
        return classes

    @staticmethod
    def make_layers(in_channels, cfg):
        layers = []
        layers += [nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=1),
                   nn.BatchNorm2d(64),
                   nn.ReLU(inplace=True)]
        in_channels = 64
        for x in cfg:
            if type(x) == str:
                x = int(x.replace('D', ''))
                layers += [CapsuleConv2d(in_channels, x, kernel_size=3, in_length=8, out_length=8, padding=1, stride=2)]
            else:
                layers += [CapsuleConv2d(in_channels, x, kernel_size=3, in_length=8, out_length=8, padding=1)]
            in_channels = x
        layers += [nn.AdaptiveAvgPool2d(4)]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    d = SquashCapsuleNet(in_channels=1, num_class=10, data_type='MNIST')
    print(d)
