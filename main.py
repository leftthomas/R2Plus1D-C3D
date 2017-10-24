import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

import model

# train_data = datasets.MNIST(root='data/MNIST', train=True, transform=transforms.ToTensor(), download=True)
# test_data = datasets.MNIST(root='data/MNIST', train=False, transform=transforms.ToTensor(), download=False)
train_data = datasets.CIFAR10(root='data/CIFAR10', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.CIFAR10(root='data/CIFAR10', train=False, transform=transforms.ToTensor(), download=False)

train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)

net = model.Net()
if torch.cuda.is_available():
    net.cuda()

criterion = nn.NLLLoss()
optimizer = optim.Adam(params=net.parameters())


for epoch in range(1, 100):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
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

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
                                                           100. * correct / len(test_loader.dataset)))
