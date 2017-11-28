import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=3):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    @staticmethod
    def squash(tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    @staticmethod
    def softmax(tensor, dim=1):
        transposed_input = tensor.transpose(dim, len(tensor.size()) - 1)
        softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(tensor.size()) - 1)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size()))
            if torch.cuda.is_available():
                logits = logits.cuda()
            for i in range(self.num_iterations):
                probs = self.softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


config = {
    'MNIST': [64, '64D', 128, '128D', 256, 256, 256, '256D', 512, 512, 512],
    'CIFAR10': [64, '64D', 128, '128D', 256, 256, 256, '256D', 512, 512, 512],
    'CIFAR100': [64, '64D', 128, '128D', 256, 256, 256, '256D', 512, 512, 512],
    'STL10': [64, '64D', 128, '128D', 256, 256, 256, '256D', 512, 512, 512],
    'SVHN': [64, '64D', 128, '128D', 256, 256, 256, '256D', 512, 512, 512],
}


class SquashCapsuleNet(nn.Module):
    def __init__(self, in_channels, num_class, data_type):
        super(SquashCapsuleNet, self).__init__()
        self.features = self.make_layers(in_channels, config[data_type])

        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=512, out_channels=32,
                                             kernel_size=1, stride=1)
        self.classifier_capsules = CapsuleLayer(num_capsules=num_class, num_route_nodes=32 * 4 * 4, in_channels=8,
                                                out_channels=16)

    def forward(self, x):
        out = self.features(x)
        out = self.primary_capsules(out)
        out = self.classifier_capsules(out).squeeze().transpose(0, 1)

        classes = (out ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)
        return classes

    @staticmethod
    def make_layers(in_channels, cfg):
        layers = []
        for x in cfg:
            if type(x) == str:
                x = int(x.replace('D', ''))
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
            in_channels = x
        layers += [nn.AdaptiveAvgPool2d(4)]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    d = SquashCapsuleNet(in_channels=1, num_class=10, data_type='MNIST')
    print(d)
