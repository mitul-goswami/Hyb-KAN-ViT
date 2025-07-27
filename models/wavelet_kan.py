import torch
import torch.nn as nn
import numpy as np
import pywt
import math

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

def dog_wavelet(x, m=2, sigma=1.0, tau=0.0):
    exponent = -((x - tau)**2) / (2 * sigma**2)
    gaussian = torch.exp(exponent)
    return (-1)**m * torch.autograd.grad(gaussian, x, grad_outputs=torch.ones_like(gaussian), create_graph=True)[0]

def mexican_hat_wavelet(x, sigma=1.0, tau=0.0):
    normalized = (x - tau) / sigma
    return (1 / torch.sqrt(sigma)) * (1 - normalized**2) * torch.exp(-normalized**2 / 2)

def morlet_wavelet(x, w0=5.0, sigma=1.0, tau=0.0):
    normalized = (x - tau) / sigma
    return torch.exp(-normalized**2 / 2) * torch.cos(w0 * normalized)

class WaveletKAN(nn.Module):
    def __init__(self, in_dim, out_dim, num_scales=6, wavelet_type='dog', initial_scale=1.0, central_freq=5.0, decomposition_levels=4, pruning_ratio=0.4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_scales = num_scales
        self.wavelet_type = wavelet_type
        self.initial_scale = initial_scale
        self.central_freq = central_freq
        self.decomposition_levels = decomposition_levels
        self.pruning_ratio = pruning_ratio
        
        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.scale_factors = nn.Parameter(torch.ones(out_dim, in_dim, num_scales))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.fwt = FastWaveletTransform(wavelet=self.get_wavelet_fn(), decomposition_levels=decomposition_levels)
        self.reset_parameters()
    
    def get_wavelet_fn(self):
        if self.wavelet_type == 'dog':
            return dog_wavelet
        elif self.wavelet_type == 'mexican_hat':
            return mexican_hat_wavelet
        elif self.wavelet_type == 'morlet':
            return lambda x: morlet_wavelet(x, self.central_freq)
        else:
            raise ValueError(f"Unknown wavelet type: {self.wavelet_type}")
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.constant_(self.scale_factors, 1.0)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        base_out = torch.matmul(x, self.base_weight.t())
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), self.in_dim, 1, 1)
        
        coeffs = self.fwt(x)
        scaled_coeffs = coeffs * self.scale_factors.unsqueeze(0)
        
        if self.pruning_ratio > 0:
            abs_coeffs = torch.abs(scaled_coeffs)
            threshold = torch.quantile(abs_coeffs, self.pruning_ratio, dim=-1, keepdim=True)
            mask = (abs_coeffs > threshold).float()
            scaled_coeffs = scaled_coeffs * mask
        
        wavelet_out = self.fwt.inverse(scaled_coeffs)
        wavelet_out = wavelet_out.view(x.size(0), self.out_dim)
        
        return base_out + wavelet_out + self.bias
