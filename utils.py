import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchnet.meter.meter import Meter
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
transform_value = {'MNIST': transforms.Normalize((0.1307,), (0.3081,)),
                   'FashionMNIST': transforms.Normalize((0.2860,), (0.3530,)),
                   'SVHN': transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
                   'CIFAR10': transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                   'CIFAR100': transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                   'STL10': transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))}


def get_iterator(mode, data_type, batch_size, use_data_augmentation):
    if use_data_augmentation:
        if data_type == 'MNIST':
            transform_train = transforms.Compose([
                transforms.RandomCrop(28, padding=2),
                transforms.ToTensor(),
                transform_value[data_type]
            ])
        elif data_type == 'FashionMNIST':
            transform_train = transforms.Compose([
                transforms.RandomCrop(28, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transform_value[data_type]
            ])
        elif data_type == 'SVHN':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=2),
                transforms.ToTensor(),
                transform_value[data_type]
            ])
        elif data_type == 'STL10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(96, padding=6),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transform_value[data_type]
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transform_value[data_type]
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transform_value[data_type]
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

    if (data_type == 'STL10') or (data_type == 'SVHN'):
        data = data_set[data_type](root='data/' + data_type, split='train' if mode else 'test',
                                   transform=transform_train if mode else transform_test, download=True)
    else:
        data = data_set[data_type](root='data/' + data_type, train=mode,
                                   transform=transform_train if mode else transform_test, download=True)

    return DataLoader(dataset=data, batch_size=batch_size, shuffle=mode, num_workers=4)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() != 2:
            raise ValueError("Expected 2D tensor as input, got {}D tensor instead.".format(input.dim()))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


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
        classes = out.norm(p=2, dim=-1)
        classes = F.softmax(classes, dim=-1)

        # if the target category equal None, return the feature map of the highest scoring category,
        # otherwise, return the feature map of the requested category
        if self.target_category is None:
            one_hot, _ = classes.max(dim=-1)
        else:
            if self.target_category > classes.size(-1) - 1:
                raise ValueError(
                    "Expected target category must less than the total categories({}).".format(classes.size(-1)))
            one_hot = classes[0][self.target_category]

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward()

        weight = self.gradients.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        cam = F.relu((weight * self.features).sum(dim=1))
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cam * 255
        img = transforms.ToPILImage()(cam.data.cpu())
        img = transforms.Resize(size=image_size)(img)
        result = transforms.ToTensor()(img)
        return result.numpy()


def get_mean_std(data_type):
    if data_type == 'MNIST' or data_type == 'FashionMNIST':
        train_set = data_set[data_type](root='data/' + data_type, train=True, download=True,
                                        transform=transforms.ToTensor())
        print(list(train_set.train_data.size()))
        print(train_set.train_data.float().mean() / 255)
        print(train_set.train_data.float().std() / 255)
    elif data_type == 'STL10':
        train_set = data_set[data_type](root='data/' + data_type, split='train', download=True,
                                        transform=transforms.ToTensor())
        print(train_set.data.shape)
        train_set.data = train_set.data.reshape((5000, 3, 96, 96))
        train_set.data = train_set.data.transpose((0, 2, 3, 1))  # convert to HWC
        print(train_set.data.mean(axis=(0, 1, 2)) / 255)
        print(train_set.data.std(axis=(0, 1, 2)) / 255)
    elif data_type == 'SVHN':
        train_set = data_set[data_type](root='data/' + data_type, split='train', download=True,
                                        transform=transforms.ToTensor())
        print(train_set.data.shape)
        train_set.data = train_set.data.reshape((73257, 3, 32, 32))
        train_set.data = train_set.data.transpose((0, 2, 3, 1))  # convert to HWC
        print(train_set.data.mean(axis=(0, 1, 2)) / 255)
        print(train_set.data.std(axis=(0, 1, 2)) / 255)
    else:
        train_set = data_set[data_type](root='data/' + data_type, train=True, download=True,
                                        transform=transforms.ToTensor())
        print(train_set.train_data.shape)
        print(train_set.train_data.mean(axis=(0, 1, 2)) / 255)
        print(train_set.train_data.std(axis=(0, 1, 2)) / 255)


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.n += n

        if self.n == 0:
            self.mean = np.nan
        elif self.n == 1:
            self.mean = self.sum
        else:
            self.mean = self.sum / self.n

    def value(self):
        return self.mean

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.val = 0.0
        self.mean = np.nan


if __name__ == "__main__":
    get_mean_std('MNIST')
    get_mean_std('FashionMNIST')
    get_mean_std('SVHN')
    get_mean_std('CIFAR10')
    get_mean_std('CIFAR100')
    get_mean_std('STL10')
    t = get_iterator(True, 'MNIST', 16, True)
    print(t)
