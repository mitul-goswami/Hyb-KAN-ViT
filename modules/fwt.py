import torch
import torch.nn as nn
import pywt
import numpy as np

class FastWaveletTransform(nn.Module):
    def __init__(self, wavelet='db1', decomposition_levels=4):
        super().__init__()
        self.wavelet = wavelet
        self.decomposition_levels = decomposition_levels
        
    def forward(self, x):
        coeffs = []
        for b in range(x.size(0)):
            channel_coeffs = []
            for c in range(x.size(1)):
                cA = x[b, c].cpu().numpy()
                level_coeffs = []
                for _ in range(self.decomposition_levels):
                    cA, (cH, cV, cD) = pywt.dwt2(cA, self.wavelet)
                    level_coeffs.append(torch.tensor(np.stack([cA, cH, cV, cD]), device=x.device))
                channel_coeffs.append(torch.stack(level_coeffs))
            coeffs.append(torch.stack(channel_coeffs))
        return torch.stack(coeffs)
    
    def inverse(self, coeffs):
        rec = []
        for b in range(coeffs.size(0)):
            channel_rec = []
            for c in range(coeffs.size(1)):
                cA = coeffs[b, c, -1, 0].cpu().numpy()
                for l in range(self.decomposition_levels-1, -1, -1):
                    cH = coeffs[b, c, l, 1].cpu().numpy()
                    cV = coeffs[b, c, l, 2].cpu().numpy()
                    cD = coeffs[b, c, l, 3].cpu().numpy()
                    cA = pywt.idwt2((cA, (cH, cV, cD)), self.wavelet)
                channel_rec.append(torch.tensor(cA, device=coeffs.device))
            rec.append(torch.stack(channel_rec))
        return torch.stack(rec)
