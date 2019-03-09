import torch
import torchvision.models as models
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


batch_size = 64
epochs = 10

cifar_10_train_dataset = CIFAR10(r'c:\data\tv', download=True, transform=ToTensor())
cifar_10_test_dataset = CIFAR10(r'c:\data\tv', download=True, transform=ToTensor(), train=False)

cifar_10_train_loader = DataLoader(cifar_10_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              pin_memory=torch.cuda.is_available())

cifar_10_test_loader = DataLoader(cifar_10_test_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              pin_memory=torch.cuda.is_available())

model = Net().to(device)
optim = Adam(model.parameters())
criterion = CrossEntropyLoss()


for epoch in range(epochs):
    train_batch = tqdm(cifar_10_train_loader, total=len(cifar_10_train_dataset) // batch_size)
    for image, target_label in train_batch:
        image, target_label = image.to(device), target_label.to(device)

        optim.zero_grad()
        label = model(image)
        loss = criterion(label, target_label)
        loss.backward()
        train_batch.set_description('Train Loss: %4f' % loss.item())
        optim.step()

    test_batch = tqdm(cifar_10_test_loader, total=len(cifar_10_test_dataset) // batch_size)
    for image, target_label in test_batch:
        image, target_label = image.to(device), target_label.to(device)

        label = model(image)
        loss = criterion(label, target_label)
        test_batch.set_description('Test Loss: %4f' % loss.item())
