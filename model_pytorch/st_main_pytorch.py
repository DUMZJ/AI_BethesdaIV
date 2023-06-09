mport torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

# Define the Swin Transformer model
class SwinTransformer(nn.Module):
    def __init__(self, num_classes=3):
        super(SwinTransformer, self).__init__()

        # Input stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Transformer layers
        self.layers = nn.Sequential(
            SwinBlock(dim=96, input_resolution=(56, 56), window_size=7, shift_size=2, depth=2),
            SwinBlock(dim=192, input_resolution=(28, 28), window_size=7, shift_size=2, depth=2),
            SwinBlock(dim=384, input_resolution=(14, 14), window_size=7, shift_size=2, depth=2),
            SwinBlock(dim=768, input_resolution=(7, 7), window_size=7, shift_size=2, depth=2)
        )

        # Output head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(768, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.head(x)
        return x

# Define the Swin Transformer block
class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, window_size, shift_size, depth):
        super(SwinBlock, self).__init__()

        self.layers = nn.ModuleList([
            SwinLayer(dim=dim, input_resolution=input_resolution, window_size=window_size, shift_size=shift_size)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

# Define the Swin Transformer layer
class SwinLayer(nn.Module):
    def __init__(self, dim, input_resolution, window_size, shift_size):
        super(SwinLayer, self).__init__()

        self.window_size = window_size
        self.shift_size = shift_size

        # Calculate number of windows
        self.num_windows = (input_resolution[0] // window_size) * (input_resolution[1] // window_size)

        # Window partitioning
        self.partition = nn.Conv2d(dim, dim, kernel_size=window_size, stride=window_size, groups=self.num_windows)

        # Absolute position encoding
        self.absolute_pos_encoding = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
# Relative position encoding
    self.relative_pos_encoding = nn.Parameter(torch.zeros(1, dim, window_size, window_size))

    # Shift operation
    self.shift = nn.ModuleList([
        nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        for _ in range(4)
    ])

    # Residual connection
    self.residual = nn.Identity()

    # Layer normalization
    self.norm1 = nn.LayerNorm(dim)

    # Feedforward network
    self.ffn = nn.Sequential(
        nn.Linear(dim, 4 * dim),
        nn.ReLU(inplace=True),
        nn.Linear(4 * dim, dim)
    )

    # Layer normalization
    self.norm2 = nn.LayerNorm(dim)

def forward(self, x):
    # Window partitioning
    x = self.partition(x)

    # Rearrange windows
    batch_size, _, height, width = x.shape
    x = x.view(batch_size, self.num_windows, -1, height, width)
    x = x.permute(0, 1, 3, 4, 2)

    # Add absolute position encoding
    x = x + self.absolute_pos_encoding

    # Add relative position encoding
    H, W = height // self.window_size, width // self.window_size
    relative_pos_encoding = self.relative_pos_encoding.repeat(1, 1, H, W)
    x = x + relative_pos_encoding

    # Shift operation
    x_top, x_bottom, x_left, x_right = torch.split(x, x.shape[1] // 4, dim=1)
    x_top_right = self.shift[0](x_top) + self.shift[1](x_right)
    x_bottom_left = self.shift[2](x_bottom) + self.shift[3](x_left)
    x = torch.cat([x_top_right, x_bottom_left], dim=1)

    # Residual connection
    res = self.residual(x)

    # Layer normalization
    x = self.norm1(x)

    # Feedforward network
    x = self.ffn(x)

    # Add residual connection
    x = x + res

    # Layer normalization
    x = self.norm2(x)

    return x