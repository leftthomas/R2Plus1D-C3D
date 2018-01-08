import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10, MNIST, FashionMNIST, STL10, SVHN

CLASS_NAME = {'MNIST': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
              'FashionMNIST': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
                               'Bag', 'Ankle boot'],
              'SVHN': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
              'CIFAR10': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
              'CIFAR100': ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
                           'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
                           'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
                           'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
                           'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                           'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
                           'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
                           'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
                           'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                           'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
                           'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
                           'willow_tree', 'wolf', 'woman', 'worm'],
              'STL10': ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']}

data_set = {'MNIST': MNIST, 'FashionMNIST': FashionMNIST, 'SVHN': SVHN, 'CIFAR10': CIFAR10, 'CIFAR100': CIFAR100,
            'STL10': STL10}


def get_iterator(mode, data_type, batch_size):
    if data_type == 'STL10' or data_type == 'SVHN':
        data = data_set[data_type](root='data/' + data_type, split='train' if mode else 'test',
                                   transform=transforms.ToTensor(), download=True)
    else:
        data = data_set[data_type](root='data/' + data_type, train=mode, transform=transforms.ToTensor(), download=True)

    return DataLoader(dataset=data, batch_size=batch_size, shuffle=mode, num_workers=4)


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def forward(self, classes, labels):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.mean()

        return margin_loss


class GradCam:
    def __init__(self, model, target_layer, target_category):
        self.model = model
        self.target_layer = len(model.features) - 1 if target_layer is None else target_layer
        if self.target_layer > len(model.features) - 1:
            raise ValueError(
                "Expected target layer must less than the total layers({}) of features.".format(len(model.features)))
        self.target_category = target_category
        self.features = None
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def __call__(self, x):
        image_size = (x.size(-2), x.size(-1))
        # save the target layer' gradients and features, then get the category scores
        for idx, module in enumerate(self.model.features.children()):
            x = module(x)
            if idx == self.target_layer:
                x.register_hook(self.save_gradient)
                self.features = x
        out = x.view(*x.size()[:2], -1)
        out = out.transpose(-1, -2)
        out = out.contiguous().view(out.size(0), -1, self.model.out_length)
        out = self.model.classifier(out)
        classes = (out ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        # if the target category equal None, return the feature map of the highest scoring category,
        # otherwise, return the feature map of the requested category
        if self.target_category is None:
            one_hot, _ = classes.max(dim=-1)
        else:
            if self.target_category > classes.size(-1) - 1:
                raise ValueError(
                    "Expected target category must less than the total categories({}).".format(classes.size(-1)))
            one_hot = classes.index_select(dim=-1, index=Variable(torch.LongTensor([self.target_category])))

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward()

        weights = self.gradients.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        cam = F.relu((weights * self.features).sum(dim=1))
        cam = cam - cam.min()
        cam = cam / cam.max()
        img = transforms.ToPILImage()(cam.data.cpu())
        img = transforms.Resize(size=image_size)(img)
        result = transforms.ToTensor()(img)
        return result
