#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/4/24 22:24
# @Author : 刘秉星
# @Site : 
# @File : main.py
# @Software: PyCharm
import torch
from torch import nn
import torch.optim as optim
from lenet5 import LeNet5
from dataset import get_data_loaders,get_data_loaders_VGG
from train_test import train, test
import matplotlib.pyplot as plt
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



'''
# 加载预训练的ResNet模型
model = models.resnet50(pretrained=True)

# 修改最后的全连接层以适应19个类别
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 19)

model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loader, test_loader = get_data_loaders('train', 'test')
'''


# 加载预训练的VGGNet模型
model = models.vgg16_bn(pretrained=True)

# 修改最后的全连接层以适应19个类别
num_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_features, 19)

model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loader, test_loader = get_data_loaders_VGG('train', 'test')


'''
使用LeNet5的

model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loader, test_loader = get_data_loaders('./train', './test')
'''


num_epochs = 50
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(model, test_loader, criterion, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

torch.save(model.state_dict(), 'vggnet_model.pth')

# 绘制准确度和损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss VGG")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Acc")
plt.plot(range(1, num_epochs + 1), test_accuracies, label="Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy VGG")
plt.legend()

plt.show()