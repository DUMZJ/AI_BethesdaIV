import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet121(nn.Module):
    def __init__(self, num_classes=3):
        super(DenseNet121, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            DenseBlock(64, 6),
            TransitionBlock(256),
            DenseBlock(128, 12),
            TransitionBlock(512),
            DenseBlock(256, 24),
            TransitionBlock(1024),
            DenseBlock(512, 16),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([DenseLayer(in_channels + i * 32) for i in range(num_layers)])

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            y = layer(torch.cat(features, 1))
            features.append(y)
        return torch.cat(features, 1)


class DenseLayer(nn.Module):
    def __init__(self, in_channels):
        super(DenseLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class TransitionBlock(nn.Module):
    def __init__(self, in_channels):
        super(TransitionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)
        return out
