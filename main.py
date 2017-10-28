import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

import model

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
# train_data = datasets.MNIST(root='data/MNIST', train=False, transform=transforms.Compose(
#     [transforms.ToTensor(), normalize]), download=True)
# test_data = datasets.MNIST(root='data/MNIST', train=False, transform=transforms.Compose(
#     [transforms.ToTensor(), normalize]), download=False)
train_data = datasets.CIFAR10(root='data/CIFAR10', train=True, transform=transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]), download=False)
test_data = datasets.CIFAR10(root='data/CIFAR10', train=False, transform=transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]), download=False)
# train_data = datasets.CIFAR100(root='data/CIFAR100', train=True, transform=transforms.Compose(
#     [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]), download=False)
# test_data = datasets.CIFAR100(root='data/CIFAR100', train=False, transform=transforms.Compose(
#     [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]), download=False)
# train_data = datasets.STL10(root='data/STL10', split='train', transform=transforms.Compose(
#     [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]), download=False)
# test_data = datasets.STL10(root='data/STL10', split='test', transform=transforms.Compose(
#     [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]), download=False)

train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=True)

net = model.Net()
if torch.cuda.is_available():
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=net.parameters(), lr=0.01, momentum=0.5, weight_decay=1e-4)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 22], gamma=0.1)
for epoch in range(1, 31):
    net.train()
    scheduler.step()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx % 10 == 0) and (batch_idx != 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))

    net.eval()
    correct = 0
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_loader.dataset),
                                                           100. * correct / len(test_loader.dataset)))
