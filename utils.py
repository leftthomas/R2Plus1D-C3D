import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
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
transform_value = {'MNIST': transforms.Normalize((0.1306604762738429,), (0.30810780717887876,)),
                   'FashionMNIST': transforms.Normalize((0.2860405969887955,), (0.35302424825650003,)),
                   'SVHN': transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                                                (0.19803012, 0.20101562, 0.19703614)),
                   'CIFAR10': transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                                   (0.24703223, 0.24348513, 0.26158784)),
                   'CIFAR100': transforms.Normalize((0.50707516, 0.48654887, 0.44091784),
                                                    (0.26733429, 0.25643846, 0.27615047)),
                   'STL10': transforms.Normalize((0.44671062, 0.43980984, 0.40664645),
                                                 (0.26034098, 0.25657727, 0.27126738))}
transform_trains = {'MNIST': transforms.Compose(
    [transforms.RandomCrop(28, padding=2), transforms.ToTensor(),
     transforms.Normalize((0.1306604762738429,), (0.30810780717887876,))]),
    'FashionMNIST': transforms.Compose(
        [transforms.RandomCrop(28, padding=2), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.2860405969887955,), (0.35302424825650003,))]),
    'SVHN': transforms.Compose([transforms.RandomCrop(32, padding=2), transforms.ToTensor(),
                                transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                                                     (0.19803012, 0.20101562, 0.19703614))]),
    'CIFAR10': transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))]),
    'CIFAR100': transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))
         ]),
    'STL10': transforms.Compose(
        [transforms.RandomCrop(96, padding=6), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.44671062, 0.43980984, 0.40664645), (0.26034098, 0.25657727, 0.27126738))])}
models = {'MNIST': MNISTCapsuleNet, 'FashionMNIST': FashionMNISTCapsuleNet, 'SVHN': SVHNCapsuleNet,
          'CIFAR10': CIFAR10CapsuleNet, 'CIFAR100': CIFAR100CapsuleNet, 'STL10': STL10CapsuleNet}


class GradCam:
    def __init__(self, model):
        self.model = model.eval()
        self.feature = None
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, x):
        image_size = (x.size(-2), x.size(-1))
        image_channel = x.size(1)
        datas = Variable(x)
        heat_maps = []
        for i in range(datas.size(0)):
            img = datas[i].data.cpu().numpy()
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)

            feature = datas[i].unsqueeze(0)
            for name, module in self.model.named_children():
                if name == 'classifier':
                    feature = feature.view(*feature.size()[:2], -1)
                    feature = feature.transpose(-1, -2)
                    feature = feature.contiguous().view(feature.size(0), -1, module[0].weight.size(-1))
                feature = module(feature)
                if name == 'features':
                    feature.register_hook(self.save_gradient)
                    self.feature = feature
            classes = feature.sum(dim=-1)
            one_hot, _ = classes.max(dim=-1)
            self.model.zero_grad()
            one_hot.backward()

            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
            mask = cv2.resize(mask.cpu().data.numpy(), image_size)
            mask = mask - np.min(mask)
            if np.max(mask) != 0:
                mask = mask / np.max(mask)
            heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = heat_map + np.float32(cv2.cvtColor(np.uint8(img.transpose((1, 2, 0)) * 255),
                                                     cv2.COLOR_RGB2BGR if image_channel == 3 else cv2.COLOR_GRAY2BGR))
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            heat_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
        heat_maps = torch.stack(heat_maps)
        return heat_maps


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
        train_set = data_set[data_type](root='data/' + data_type, train=True, download=True)
        print(list(train_set.train_data.size()))
        print(train_set.train_data.float().mean() / 255)
        print(train_set.train_data.float().std() / 255)
    elif data_type == 'STL10':
        train_set = data_set[data_type](root='data/' + data_type, split='train', download=True)
        print(train_set.data.shape)
        train_set.data = train_set.data.transpose((0, 2, 3, 1))  # convert to HWC
        print(train_set.data.mean(axis=(0, 1, 2)) / 255)
        print(train_set.data.std(axis=(0, 1, 2)) / 255)
    elif data_type == 'SVHN':
        train_set = data_set[data_type](root='data/' + data_type, split='train', download=True)
        print(train_set.data.shape)
        train_set.data = train_set.data.transpose((0, 2, 3, 1))  # convert to HWC
        print(train_set.data.mean(axis=(0, 1, 2)) / 255)
        print(train_set.data.std(axis=(0, 1, 2)) / 255)
    else:
        train_set = data_set[data_type](root='data/' + data_type, train=True, download=True)
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
