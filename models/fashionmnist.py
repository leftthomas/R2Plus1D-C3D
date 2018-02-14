from torch import nn

from capsulelayer import CapsuleConv2d, CapsuleLinear


class FashionMNISTCapsuleNet(nn.Module):
    def __init__(self, with_conv_routing=False, with_linear_routing=False):
        super(FashionMNISTCapsuleNet, self).__init__()
        self.features_out_length = 8
        self.features = nn.Sequential(
            CapsuleConv2d(in_channels=1, out_channels=32, kernel_size=9, in_length=1, out_length=4, stride=1,
                          padding=0, with_routing=with_conv_routing),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            CapsuleConv2d(in_channels=32, out_channels=64, kernel_size=9, in_length=4,
                          out_length=self.features_out_length, stride=2, padding=0, with_routing=with_conv_routing),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.classifier = CapsuleLinear(in_capsules=4 * 4 * 64 // self.features_out_length, out_capsules=10,
                                        in_length=self.features_out_length, out_length=16,
                                        with_routing=with_linear_routing)

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, self.features_out_length)

        out = self.classifier(out)
        classes = out.sum(dim=-1)
        return classes
