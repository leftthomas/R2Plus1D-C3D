from capsule_layer import CapsuleLinear
from torch import nn


class MNISTCapsuleNet(nn.Module):
    def __init__(self, num_iterations=3):
        super(MNISTCapsuleNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.features = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                      nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU())
        self.classifier = nn.Sequential(CapsuleLinear(out_capsules=10, in_length=128, out_length=16, in_capsules=None,
                                                      share_weight=True, routing_type='contract',
                                                      num_iterations=num_iterations))

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)

        out = out.permute(0, 2, 3, 1)
        out = out.contiguous().view(out.size(0), -1, 128)

        out = self.classifier(out)
        classes = out.norm(dim=-1)
        return classes


if __name__ == '__main__':
    model = MNISTCapsuleNet()
    for m in model.named_children():
        print(m)
