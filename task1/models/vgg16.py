import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg import init_weights_ # Assuming utils/nn.py exists as before

class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers, BatchNorm, and ReLU.
    Includes optional downsampling via stride or 1x1 convolution.
    """
    expansion = 1 # For basic blocks, input channels == output channels

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

        # Skip connection (shortcut)
        self.shortcut = nn.Sequential()
        # If stride > 1 (downsampling) or in_channels != out_channels*expansion,
        # we need to project the shortcut connection.
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x # Store the input for the shortcut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity) # Add the shortcut connection
        out = self.relu(out) # Final ReLU after addition
        return out

class ResVGG16(nn.Module):
    """
    A VGG16-inspired model using Residual Blocks, adapted for CIFAR-10 (32x32 input).
    Includes BatchNorm (within blocks) and Dropout (in classifier).
    Fulfills Task 1 requirements: Conv2D, Pool, FC, Activation, BatchNorm, Dropout, Residual Connection.
    """
    def __init__(self, block=ResidualBlock, num_blocks_list=[2, 2, 3, 3, 3], num_classes=10, dropout_p=0.5):
        super(ResVGG16, self).__init__()
        self.in_channels = 64

        # Initial convolution (similar to ResNet) - Input 3x32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # After conv1: 64x32x32

        # VGG-like stages using Residual Blocks
        self.layer1 = self._make_layer(block, 64, num_blocks_list[0], stride=1)   # Output: 64x32x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                      # Output: 64x16x16

        self.layer2 = self._make_layer(block, 128, num_blocks_list[1], stride=1) # Output: 128x16x16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                      # Output: 128x8x8

        self.layer3 = self._make_layer(block, 256, num_blocks_list[2], stride=1) # Output: 256x8x8
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)                      # Output: 256x4x4

        self.layer4 = self._make_layer(block, 512, num_blocks_list[3], stride=1) # Output: 512x4x4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)                      # Output: 512x2x2

        self.layer5 = self._make_layer(block, 512, num_blocks_list[4], stride=1) # Output: 512x2x2
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)                      # Output: 512x1x1

        # Adaptive pooling to ensure 1x1 spatial size before FC layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier with Dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * block.expansion, 512), # Adjust input size based on expansion factor
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p), # Dropout layer 1
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p), # Dropout layer 2
            nn.Linear(512, num_classes)
        )

        # Initialize weights
        self.apply(init_weights_) # Use the same initialization as VGG_A

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # The first block in a layer might handle downsampling (stride > 1)
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for current_stride in strides:
            layers.append(block(self.in_channels, out_channels, current_stride))
            self.in_channels = out_channels * block.expansion # Update in_channels for the next block
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.pool1(self.layer1(out))
        out = self.pool2(self.layer2(out))
        out = self.pool3(self.layer3(out))
        out = self.pool4(self.layer4(out))
        out = self.pool5(self.layer5(out))
        out = self.avgpool(out)
        # out = torch.flatten(out, 1) # Flatten handled in classifier
        out = self.classifier(out)
        return out

# Example of how to instantiate:
# model = ResVGG16(num_classes=10, dropout_p=0.5)