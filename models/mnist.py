from torch import nn

from capsulelayer import CapsuleConv2d, CapsuleLinear


class MNISTCapsuleNet(nn.Module):
    def __init__(self):
        super(MNISTCapsuleNet, self).__init__()
        self.out_length = 8
        self.features = nn.Sequential(
            CapsuleConv2d(in_channels=1, out_channels=16, kernel_size=5, in_length=1, out_length=4, stride=1,
                          padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            CapsuleConv2d(in_channels=16, out_channels=32, kernel_size=3, in_length=4, out_length=4, stride=2,
                          padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            CapsuleConv2d(in_channels=32, out_channels=32, kernel_size=3, in_length=4, out_length=self.out_length,
                          stride=2,
                          padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )
        self.classifier = CapsuleLinear(in_capsules=7 * 7 * 32 // self.out_length, out_capsules=10,
                                        in_length=self.out_length, out_length=self.out_length)

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, self.out_length)

        out = self.classifier(out)
        classes = out.sum(dim=-1)
        return classes
