from capsule_layer import CapsuleLinear
from torch import nn


class MNISTCapsuleNet(nn.Module):
    def __init__(self, routing_type='sum'):
        super(MNISTCapsuleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            CapsuleLinear(in_capsules=512, out_capsules=128, in_length=4, out_length=8, routing_type=routing_type,
                          share_weight=True),
            CapsuleLinear(in_capsules=128, out_capsules=10, in_length=8, out_length=16, routing_type=routing_type,
                          share_weight=True))

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, 4)

        out = self.classifier(out)
        classes = out.norm(dim=-1)
        return classes
