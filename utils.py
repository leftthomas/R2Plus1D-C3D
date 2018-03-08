import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10, MNIST, FashionMNIST, STL10, SVHN

from models.cifar10 import CIFAR10CapsuleNet
from models.cifar100 import CIFAR100CapsuleNet
from models.fashionmnist import FashionMNISTCapsuleNet
from models.mnist import MNISTCapsuleNet
from models.stl10 import STL10CapsuleNet
from models.svhn import SVHNCapsuleNet

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
transform_trains = {'MNIST': transforms.Compose(
    [transforms.RandomCrop(28, padding=2), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    'FashionMNIST': transforms.Compose(
        [transforms.RandomCrop(28, padding=2), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.2860,), (0.3530,))]),
    'SVHN': transforms.Compose([transforms.RandomCrop(32, padding=2), transforms.ToTensor(),
                                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
    'CIFAR10': transforms.Compose(
        [transforms.RandomCrop(32, padding=2), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
    'CIFAR100': transforms.Compose(
        [transforms.RandomCrop(32, padding=2), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
         ]),
    'STL10': transforms.Compose(
        [transforms.RandomCrop(96, padding=6), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))])}
models = {'MNIST': MNISTCapsuleNet, 'FashionMNIST': FashionMNISTCapsuleNet, 'SVHN': SVHNCapsuleNet,
          'CIFAR10': CIFAR10CapsuleNet, 'CIFAR100': CIFAR100CapsuleNet, 'STL10': STL10CapsuleNet}


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, classes, labels):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        loss = labels * left + 0.25 * (1 - labels) * right
        return loss.mean()


def show_features(model, target_layer, data):
    model = model.eval()
    target_layer = len(model.features) - 1 if target_layer is None else target_layer
    if target_layer > len(model.features) - 1:
        raise ValueError("Expected target layer must less than the total layers({}) "
                         "of features.".format(len(model.features)))
    for idx, module in enumerate(model.features.children()):
        data = module(data)
        if idx == target_layer:
            features = data
    return features.means(dim=1, keepdim=True).data.cpu()


def get_iterator(mode, data_type, batch_size=64, use_data_augmentation=True):
    if use_data_augmentation:
        transform_train = transform_trains[data_type]
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


if __name__ == "__main__":
    get_mean_std('MNIST')
    get_mean_std('FashionMNIST')
    get_mean_std('SVHN')
    get_mean_std('CIFAR10')
    get_mean_std('CIFAR100')
    get_mean_std('STL10')
    t = get_iterator(True, 'MNIST', 16, True)
    print(t)
