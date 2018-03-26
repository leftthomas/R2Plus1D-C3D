from capsule_layer import CapsuleLinear
from torch import nn

from densenet import densenet


class CIFAR100CapsuleNet(nn.Module):
    def __init__(self, num_iterations=3):
        super(CIFAR100CapsuleNet, self).__init__()

        layers = []
        for name, module in densenet(depth=100, k=12).named_children():
            if isinstance(module, nn.AvgPool2d) or isinstance(module, nn.Linear):
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.classifier = nn.Sequential(CapsuleLinear(in_capsules=114, out_capsules=100, in_length=3, out_length=8,
                                                      routing_type='contract', share_weight=False,
                                                      num_iterations=num_iterations))

    def forward(self, x):
        out = self.features(x)
        out = self.pool(out)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, 3)

        out = self.classifier(out)
        classes = out.sum(dim=-1)
        return classes


if __name__ == '__main__':
    model = CIFAR100CapsuleNet()
    for m in model.named_children():
        print(m)
