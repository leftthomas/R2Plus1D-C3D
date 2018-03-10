from capsule_layer import CapsuleConv2d, CapsuleLinear
from torch import nn


class SVHNCapsuleNet(nn.Module):
    def __init__(self, routing_type='sum', num_iterations=3):
        super(SVHNCapsuleNet, self).__init__()
        self.features = nn.Sequential(
            CapsuleConv2d(in_channels=3, out_channels=16, kernel_size=5, in_length=3, out_length=4, stride=1,
                          padding=2),
            CapsuleConv2d(in_channels=16, out_channels=16, kernel_size=3, in_length=4, out_length=4, stride=2,
                          padding=1),
            CapsuleConv2d(in_channels=16, out_channels=32, kernel_size=3, in_length=4, out_length=8, stride=1,
                          padding=1),
            CapsuleConv2d(in_channels=32, out_channels=32, kernel_size=3, in_length=8, out_length=8, stride=2,
                          padding=1),
            CapsuleConv2d(in_channels=32, out_channels=64, kernel_size=3, in_length=8, out_length=16, stride=1,
                          padding=1),
            CapsuleConv2d(in_channels=64, out_channels=64, kernel_size=3, in_length=16, out_length=16, stride=2,
                          padding=1)
        )
        self.classifier = CapsuleLinear(in_capsules=64, out_capsules=10, in_length=16, out_length=32,
                                        routing_type=routing_type, num_iterations=num_iterations)

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, 16)

        out = self.classifier(out)
        classes = out.norm(dim=-1)
        return classes
