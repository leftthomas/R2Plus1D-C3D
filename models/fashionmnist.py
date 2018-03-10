from capsule_layer import CapsuleLinear
from torch import nn


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride == 2:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.out_channels),
            )
            residual = downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FashionMNISTCapsuleNet(nn.Module):
    def __init__(self, routing_type='sum', num_iterations=3):
        super(FashionMNISTCapsuleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            BasicBlock(in_channels=64, out_channels=64),
            BasicBlock(in_channels=64, out_channels=128, stride=2),
            BasicBlock(in_channels=128, out_channels=256, stride=2),
            BasicBlock(in_channels=256, out_channels=512, stride=2)
        )
        self.classifier = CapsuleLinear(in_capsules=256, out_capsules=10, in_length=8, out_length=16,
                                        routing_type=routing_type, share_weight=False, num_iterations=num_iterations)

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, 8)

        out = self.classifier(out)
        classes = out.norm(dim=-1)
        return classes
