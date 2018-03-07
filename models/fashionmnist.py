from capsule_layer import CapsuleConv2d, CapsuleLinear
from torch import nn


class FashionMNISTCapsuleNet(nn.Module):
    def __init__(self, routing_type='sum'):
        super(FashionMNISTCapsuleNet, self).__init__()
        self.features_out_length = 8
        self.features = nn.Sequential(
            CapsuleConv2d(in_channels=1, out_channels=16, kernel_size=5, in_length=1, out_length=4, stride=2,
                          padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),

            CapsuleConv2d(in_channels=16, out_channels=32, kernel_size=3, in_length=4,
                          out_length=self.features_out_length, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            CapsuleLinear(in_capsules=7 * 7 * 32 // self.features_out_length, out_capsules=64,
                          in_length=self.features_out_length, out_length=16, routing_type=routing_type),
            CapsuleLinear(in_capsules=64, out_capsules=10, in_length=16, out_length=32, routing_type=routing_type))

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, self.features_out_length)

        out = self.classifier(out)
        classes = out.norm(dim=-1)
        return classes
