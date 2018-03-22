from capsule_layer import CapsuleLinear
from torch import nn
from torchvision.models.resnet import resnet18


class FashionMNISTCapsuleNet(nn.Module):
    def __init__(self, num_iterations=3):
        super(FashionMNISTCapsuleNet, self).__init__()

        layers = []
        for name, module in resnet18().named_children():
            if isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d) or isinstance(module, nn.Linear):
                continue
            if name == 'conv1':
                module.in_channels = 1
            layers.append(module)
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(CapsuleLinear(in_capsules=512, out_capsules=128, in_length=4, out_length=8,
                                                      routing_type='contract', share_weight=True,
                                                      num_iterations=num_iterations),
                                        CapsuleLinear(in_capsules=128, out_capsules=10, in_length=8, out_length=16,
                                                      routing_type='contract', share_weight=False,
                                                      num_iterations=num_iterations))

    def forward(self, x):
        out = self.features(x)

        out = out.view(*out.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, 4)

        out = self.classifier(out)
        classes = out.norm(dim=-1)
        return classes


if __name__ == '__main__':
    model = FashionMNISTCapsuleNet()
    for module in model.named_children():
        print(module)
