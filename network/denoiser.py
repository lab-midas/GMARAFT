import torch
import torch.nn as nn


class resBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(resBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_planes, planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = x
        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y))
        x = self.conv3(x)
        return self.relu(x + y)

class ResNet(nn.Module):
    def __init__(self, in_channels=3, num_filters=64):
        super(ResNet, self).__init__()
        self.num_filters = num_filters
        self.relu = nn.ReLU(inplace=True)
        self.resblock1 = resBlock(in_channels, self.num_filters)
        self.resblock2 = resBlock(self.num_filters, self.num_filters)
        self.resblock3 = resBlock(self.num_filters, self.num_filters)
        self.resblock4 = resBlock(self.num_filters, self.num_filters)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.num_filters, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.resblock1(y)
        y = self.resblock2(y)
        y = self.resblock3(y)
        y = self.resblock4(y)
        y = self.conv2(y)
        return x + y