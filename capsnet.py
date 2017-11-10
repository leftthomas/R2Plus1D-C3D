import torch.nn.functional as F
from torch import nn

import config
from capsule import CapsuleLayer


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=64,
                                             kernel_size=9, stride=2)
        self.class_capsules = CapsuleLayer(num_capsules=config.NUM_CLASSES, num_route_nodes=32 * 4 * 4,
                                           in_channels=8,
                                           out_channels=16)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.class_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)
        return classes


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def forward(self, classes, labels):
        return F.cross_entropy(classes, labels)


if __name__ == "__main__":
    model = CapsuleNet()
    print(model)
