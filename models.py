import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score
from sklearn.utils import class_weight
from sklearn import preprocessing
import numpy as np
import joblib
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
# Model definition
class ODConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding='same', groups=1, reduction=0.0625, kernel_num=4):
        super(ODConv, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2 if padding == 'same' else 0
        self.groups = groups
        self.attention_channel = max(int(in_planes * reduction), 16)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, self.attention_channel, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.attention_channel)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(self.attention_channel, in_planes, kernel_size=1, padding=0, bias=True)
        self.kernel_weights = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.kernel_weights, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        attention = self.avgpool(x)
        attention = self.fc1(attention)
        attention = self.bn1(attention)
        attention = self.relu(attention)
        channel_attention = torch.sigmoid(self.fc2(attention))
        x = x * channel_attention
        aggregate_weight = torch.sum(self.kernel_weights, dim=0)
        output = nn.functional.conv2d(x, aggregate_weight, stride=self.stride, padding=self.padding, groups=self.groups)
        return output

class GNConv(nn.Module):
    def __init__(self, in_channels, dim, order=3, kernel_size=7):
        super(GNConv, self).__init__()
        self.order = order
        self.kernel_size = kernel_size
        self.conv_in = nn.Conv2d(in_channels, dim * 2, kernel_size=1, padding=0)
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.conv_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.scale = 1.0

    def forward(self, x):
        x_proj = self.conv_in(x)
        pwa, abc = torch.split(x_proj, x_proj.shape[1] // 2, dim=1)
        dw_abc = self.depthwise_conv(abc) * self.scale
        x = pwa * dw_abc
        x = self.conv_out(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualSEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class BuildModel(nn.Module):
    def __init__(self, input_shape=(2, 80, 80), num_classes=15):
        super(BuildModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.odconv1 = ODConv(in_planes=16, out_planes=32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.res_se_block1 = ResidualSEBlock(32, 64, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.gnconv = GNConv(in_channels=64, dim=128)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.res_se_block2 = ResidualSEBlock(128, 256, stride=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout2 = nn.Dropout(0.4)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 5 * 5, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.odconv1(x)
        x = self.bn2(x)
        x = self.maxpool1(x)
        x = self.res_se_block1(x)
        x = self.maxpool2(x)
        x = self.gnconv(x)
        x = self.bn3(x)
        x = self.dropout1(x)
        x = self.maxpool3(x)
        x = self.res_se_block2(x)
        x = self.bn4(x)
        x = self.dropout2(x)
        x = self.maxpool4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn5(x)
        x = self.dropout3(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x