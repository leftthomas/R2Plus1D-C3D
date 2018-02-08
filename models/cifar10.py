from torch import nn


class CIFAR10CapsuleNet(nn.Module):
    def __init__(self, with_conv_routing=False, with_linear_routing=False):
        super(CIFAR10CapsuleNet, self).__init__()
        self.out_length = 1
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(in_features=8 * 8 * 128, out_features=10)

    def forward(self, x):
        out = self.features(x)

        # out = out.view(*out.size()[:2], -1)
        # out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1)

        classes = self.classifier(out)
        # classes = out.sum(dim=-1)
        return classes
