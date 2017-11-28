from torch import nn

config = {
    'MNIST': [64, '64D', 128, '128D', 256, 256, 256, '256D', 512, 512, 512, '512D'],
    'CIFAR10': [64, '64D', 128, '128D', 256, 256, 256, '256D', 512, 512, 512, '512D'],
    'CIFAR100': [64, '64D', 128, '128D', 256, 256, 256, '256D', 512, 512, 512, '512D'],
    'STL10': [64, '64D', 128, '128D', 256, 256, 256, '256D', 512, 512, 512, '512D'],
    'SVHN': [64, '64D', 128, '128D', 256, 256, 256, '256D', 512, 512, 512, '512D'],
}


class SquashCapsuleNet(nn.Module):
    def __init__(self, in_channels, num_class, data_type):
        super(SquashCapsuleNet, self).__init__()
        self.features = self.make_layers(in_channels, config[data_type])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_class),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

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
        layers += [nn.AdaptiveAvgPool2d(2)]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    d = SquashCapsuleNet(in_channels=1, num_class=10, data_type='MNIST')
    print(d)
