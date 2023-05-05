import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionV4(nn.Module):
    def __init__(self, num_classes=3):
        super(InceptionV4, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Inception A
        self.inception_a1 = InceptionA(64, 32, 32, 32, 32, 48, 64)
        self.inception_a2 = InceptionA(256, 64, 64, 64, 64, 96, 96)
        self.inception_a3 = InceptionA(288, 64, 64, 64, 64, 96, 96)
        
        # Reduction A
        self.reduction_a = ReductionA(288, 384, 192, 224, 256)
        
        # Inception B
        self.inception_b1 = InceptionB(1024, 192, 128, 192, 192, 256, 320)
        self.inception_b2 = InceptionB(1536, 256, 160, 256, 256, 384, 384)
        self.inception_b3 = InceptionB(2048, 256, 160, 256, 256, 384, 384)
        
        # Reduction B
        self.reduction_b = ReductionB(2048, 192, 320, 192, 192)
        
        # Classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a1(x)
        x = self.inception_a2(x)
        x = self.inception_a3(x)
        x = self.reduction_a(x)
        x = self.inception_b1(x)
        x = self.inception_b2(x)
        x = self.inception_b3(x)
        x = self.reduction_b(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_channels, conv1x1_channels, conv3x3_reduce_channels, conv3x3_channels, conv5x5_reduce_channels, conv5x5_channels):
        super(InceptionA, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            nn.Conv2d(in_channels, pool_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(pool_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, conv1x1_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(conv1x1_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, conv3x3_reduce_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(conv3x3_reduce_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv3x3_reduce_channels, conv3x3_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv3x3_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, conv5x5_reduce_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(conv5x5_reduce_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv5x5_reduce_channels, conv5x5_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(conv5x5_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class ReductionA(nn.Module):
    def __init__(self, in_channels, conv3x3_reduce_channels, conv3x3_channels, conv5x5_reduce_channels, conv5x5_channels, pool_channels):
        super(ReductionA, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, conv3x3_reduce_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(conv3x3_reduce_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv3x3_reduce_channels, conv3x3_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv3x3_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, conv5x5_reduce_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(conv5x5_reduce_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv5x5_reduce_channels, conv5x5_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(conv5x5_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        x = Concatenate(axis=3)(outputs)
        x = self.conv(x)
        x = self.bn(x)
        x = Activation('relu')(x)
        return x
        
def inception_v4_block(x, scale=1.0, activation='relu'):
    """
    Inception v4 block implementation
    """
    # Branch 1
    branch_1 = conv2d_bn(x, 64, 1, 1, activation=activation)

    # Branch 2
    branch_2 = conv2d_bn(x, 64, 1, 1, activation=activation)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3, activation=activation)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3, activation=activation)

    # Branch 3
    branch_3 = conv2d_bn(x, 64, 1, 1, activation=activation)
    branch_3 = conv2d_bn(branch_3, 96, 3, 3, activation=activation)

    # Branch 4
    branch_4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_4 = conv2d_bn(branch_4, 64, 1, 1, activation=activation)

    # Concatenate branches
    x = concatenate([branch_1, branch_2, branch_3, branch_4], axis=3)

    # Scale output
    x = Lambda(lambda t: t * scale)(x)

    return x