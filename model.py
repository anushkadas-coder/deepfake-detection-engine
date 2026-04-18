import torch
import torch.nn as nn
from torchvision import models

def get_fft_cnn():
    # Load a pre-trained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Modify the first convolutional layer to take 1 channel (grayscale FFT) instead of 3 (RGB)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify the final Fully Connected layer for binary classification (Real vs Fake)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid() # Output a probability between 0 and 1
    )
    
    return model