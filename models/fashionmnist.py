from capsule_layer import CapsuleConv2d, CapsuleLinear
from torch import nn


class FashionMNISTCapsuleNet(nn.Module):
    def __init__(self, routing_type='sum'):
        super(FashionMNISTCapsuleNet, self).__init__()
        self.features_out_length = 32
        self.features = nn.Sequential(
            CapsuleConv2d(in_channels=1, out_channels=64, kernel_size=3, in_length=1, out_length=4, stride=1,
                          padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            CapsuleConv2d(in_channels=64, out_channels=64, kernel_size=3, in_length=4, out_length=4, stride=2,
                          padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            CapsuleConv2d(in_channels=64, out_channels=128, kernel_size=3, in_length=4, out_length=8, stride=1,
                          padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            CapsuleConv2d(in_channels=128, out_channels=128, kernel_size=3, in_length=8, out_length=8, stride=2,
                          padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            CapsuleConv2d(in_channels=128, out_channels=256, kernel_size=3, in_length=8, out_length=16, stride=1,
                          padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            CapsuleConv2d(in_channels=256, out_channels=256, kernel_size=3, in_length=16, out_length=16, stride=1,
                          padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            CapsuleConv2d(in_channels=256, out_channels=256, kernel_size=3, in_length=16, out_length=16, stride=2,
                          padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            CapsuleConv2d(in_channels=256, out_channels=512, kernel_size=3, in_length=16, out_length=32, stride=1,
                          padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            CapsuleConv2d(in_channels=512, out_channels=512, kernel_size=3, in_length=32, out_length=32, stride=1,
                          padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            CapsuleConv2d(in_channels=512, out_channels=512, kernel_size=3, in_length=32,
                          out_length=self.features_out_length, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )
        self.classifier = CapsuleLinear(in_capsules=2 * 2 * 512 // self.features_out_length, out_capsules=10,
                                        in_length=self.features_out_length, out_length=36, routing_type=routing_type,
                                        share_weight=False)

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, self.features_out_length)

        out = self.classifier(out)
        classes = out.norm(dim=-1)
        return classes
