import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficient_kan import EfficientKAN
from .wavelet_kan import WaveletKAN

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, wavelet_type='dog'):
        super().__init__()
        self.bbox_head = nn.Sequential(
            EfficientKAN(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, num_classes * 4)
        )
        self.mask_head = nn.Sequential(
            WaveletKAN(in_channels, in_channels, wavelet_type=wavelet_type),
            nn.ReLU(),
            nn.Linear(in_channels, num_classes)
        )
    
    def forward(self, x):
        bbox_pred = self.bbox_head(x)
        mask_pred = self.mask_head(x)
        return bbox_pred, mask_pred

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, wavelet_type='dog'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
        self.kan1 = WaveletKAN(in_channels // 2, in_channels // 2, wavelet_type=wavelet_type)
        self.conv2 = nn.Conv2d(in_channels // 2, num_classes, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
        x = self.kan1(x)
        x = x.view(b, h, w, -1).permute(0, 3, 1, 2)
        x = self.conv2(x)
        return x
