import torch.nn.functional as F
from torch import nn

from capsulelayer import CapsuleConv2d, CapsuleLinear


class FashionMNISTCapsuleNet(nn.Module):
    def __init__(self):
        super(FashionMNISTCapsuleNet, self).__init__()
        self.out_length = 8
        self.features = nn.Sequential(
            CapsuleConv2d(in_channels=1, out_channels=32, kernel_size=7, in_length=1, out_length=4, stride=2,
                          padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            CapsuleConv2d(in_channels=32, out_channels=64, kernel_size=3, in_length=8, out_length=self.out_length,
                          stride=2,
                          padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(CapsuleLinear(in_capsules=5 * 5 * 64 // self.out_length, out_capsules=10,
                                                      in_length=self.out_length, out_length=self.out_length),
                                        nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, self.out_length)

        out = self.classifier(out)
        classes = out.norm(p=2, dim=-1)
        classes = F.softmax(classes, dim=-1)
        return classes
