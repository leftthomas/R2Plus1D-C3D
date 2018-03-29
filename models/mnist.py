from capsule_layer import CapsuleLinear
from torch import nn

from resnet import resnet20


class MNISTCapsuleNet(nn.Module):
    def __init__(self, num_iterations=3):
        super(MNISTCapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        layers = []
        for name, module in resnet20().named_children():
            if name == 'conv1' or isinstance(module, nn.AvgPool2d) or isinstance(module, nn.Linear):
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(CapsuleLinear(out_capsules=10, in_length=64, out_length=16, in_capsules=None,
                                                      share_weight=False, routing_type='contract',
                                                      num_iterations=num_iterations))

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)

        out = out.permute(0, 2, 3, 1)
        out = out.contiguous().view(out.size(0), -1, 64)

        out = self.classifier(out)
        classes = out.norm(dim=-1)
        return classes


if __name__ == '__main__':
    model = MNISTCapsuleNet()
    for m in model.named_children():
        print(m)
