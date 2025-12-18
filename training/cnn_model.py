"""
Lightweight CNN architecture for training on selected subsets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class LightweightCNN(nn.Module):
    """
    Lightweight CNN for MNIST classification.
    
    Architecture:
    - 3 Conv blocks (32, 64, 128 filters)
    - Each block: Conv -> BatchNorm -> ReLU -> MaxPool
    - Flatten
    - Dense layer (128 units) with Dropout
    - Output layer (num_classes)
    """
    
    def __init__(self, num_classes=10, channels=None, dense_units=None, dropout=None):
        """
        Initialize CNN.
        
        Args:
            num_classes: Number of output classes
            channels: List of channel sizes for conv blocks. If None, uses config.
            dense_units: Number of units in dense layer. If None, uses config.
            dropout: Dropout probability. If None, uses config.
        """
        super(LightweightCNN, self).__init__()
        
        if channels is None:
            channels = config.CNN_CHANNELS
        if dense_units is None:
            dense_units = config.CNN_DENSE_UNITS
        if dropout is None:
            dropout = config.CNN_DROPOUT
        
        # Conv blocks
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size (MNIST: 28x28 -> 14x14 -> 7x7 -> 3x3 after 3 pools)
        # Actually: 28 -> 14 -> 7 -> 3 (with padding=1, kernel=3, stride=2)
        self.flatten_size = channels[2] * 3 * 3
        
        # Dense layers
        self.fc1 = nn.Linear(self.flatten_size, dense_units)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_units, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        # Conv block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        # Use flatten to handle channels_last/non-contiguous tensors safely
        x = torch.flatten(x, 1)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def create_cnn(num_classes=None, **kwargs):
    """
    Create a lightweight CNN model.
    
    Args:
        num_classes: Number of output classes. If None, uses config.NUM_CLASSES.
        **kwargs: Additional arguments passed to LightweightCNN
        
    Returns:
        Initialized CNN model
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    
    model = LightweightCNN(num_classes=num_classes, **kwargs)
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing CNN model creation...")
    
    model = create_cnn()
    print(f"Model created: {model}")
    
    # Test forward pass
    x = torch.randn(4, 1, 28, 28)  # Batch of 4 MNIST images
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

