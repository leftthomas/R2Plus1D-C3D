import torch.nn.functional as F
from torch import nn

from capsulelayer import CapsuleLinear

config = {
    'MNIST': [64, '64D', 128, '128D', 256, 256, 256, '256D', 512, 512, 512],
    'CIFAR10': [64, '64D', 128, '128D', 256, 256, 256, '256D', 512, 512, 512],
    'CIFAR100': [64, '64D', 128, '128D', 256, 256, 256, '256D', 512, 512, 512],
    'STL10': [64, '64D', 128, '128D', 256, 256, 256, '256D', 512, 512, 512],
    'SVHN': [64, '64D', 128, '128D', 256, 256, 256, '256D', 512, 512, 512],
}


class SquashCapsuleNet(nn.Module):
    def __init__(self, in_channels, num_class, data_type):
        super(SquashCapsuleNet, self).__init__()
        self.features = self.make_layers(in_channels, config[data_type])
        self.classifier = CapsuleLinear(num_capsules=num_class, num_route_nodes=32 * 4 * 4, in_channels=8,
                                        out_channels=16)

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)

        classes = (out ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)
        return classes

    @staticmethod
    def make_layers(in_channels, cfg):
        layers = []
        for x in cfg:
            if type(x) == str:
                x = int(x.replace('D', ''))
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
            in_channels = x
        layers += [nn.AdaptiveAvgPool2d(4)]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    d = SquashCapsuleNet(in_channels=1, num_class=10, data_type='MNIST')
    print(d)
