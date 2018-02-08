from torch import nn

from capsulelayer import CapsuleConv2d, CapsuleLinear


class CIFAR100CapsuleNet(nn.Module):
    def __init__(self, with_conv_routing=False, with_linear_routing=False):
        super(CIFAR100CapsuleNet, self).__init__()
        self.out_length = 16
        self.features = nn.Sequential(
            CapsuleConv2d(in_channels=3, out_channels=16, kernel_size=5, in_length=1, out_length=4, stride=1,
                          padding=2, with_routing=with_conv_routing),
            CapsuleConv2d(in_channels=16, out_channels=16, kernel_size=5, in_length=4, out_length=4, stride=2,
                          padding=2, with_routing=with_conv_routing),
            CapsuleConv2d(in_channels=16, out_channels=32, kernel_size=3, in_length=4, out_length=8, stride=1,
                          padding=1, with_routing=with_conv_routing),
            CapsuleConv2d(in_channels=32, out_channels=32, kernel_size=3, in_length=8, out_length=8, stride=2,
                          padding=1, with_routing=with_conv_routing),
            CapsuleConv2d(in_channels=32, out_channels=64, kernel_size=3, in_length=8, out_length=16, stride=1,
                          padding=1, with_routing=with_conv_routing),
            CapsuleConv2d(in_channels=64, out_channels=64, kernel_size=3, in_length=16, out_length=16, stride=1,
                          padding=1, with_routing=with_conv_routing),
            CapsuleConv2d(in_channels=64, out_channels=128, kernel_size=3, in_length=16, out_length=self.out_length,
                          stride=2, padding=1, with_routing=with_conv_routing)
        )
        self.classifier = CapsuleLinear(in_capsules=4 * 4 * 128 // self.out_length, out_capsules=100,
                                        in_length=self.out_length, out_length=self.out_length,
                                        with_routing=with_linear_routing)

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, self.out_length)

        out = self.classifier(out)
        classes = out.sum(dim=-1)
        return classes
