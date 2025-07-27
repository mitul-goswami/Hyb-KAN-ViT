import torch
import numpy as np

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

def fwt_1d(x, wavelet, scales, dx=0.1):
    results = []
    for scale in scales:
        t = torch.arange(-len(x)//2, len(x)//2) * dx
        psi = wavelet(t/scale)
        psi = psi / torch.sqrt(scale)
        result = torch.conv1d(x.view(1,1,-1), psi.view(1,1,-1), padding='same')
        results.append(result.squeeze())
    return torch.stack(results, dim=-1)

def get_wavelet_gradients(x, wavelet_type, sigma, tau, w0=None):
    x = x.detach().requires_grad_(True)
    if wavelet_type == 'dog':
        w = dog_wavelet(x, m=2, sigma=sigma, tau=tau)
    elif wavelet_type == 'mexican_hat':
        w = mexican_hat_wavelet(x, sigma=sigma, tau=tau)
    elif wavelet_type == 'morlet':
        w = morlet_wavelet(x, w0=w0, sigma=sigma, tau=tau)
    else:
        raise ValueError(f"Unknown wavelet type: {wavelet_type}")
    dw_dsigma = torch.autograd.grad(w, sigma, grad_outputs=torch.ones_like(w), retain_graph=True)[0]
    if wavelet_type == 'morlet':
        dw_dw0 = torch.autograd.grad(w, w0, grad_outputs=torch.ones_like(w), retain_graph=True)[0]
        return w, dw_dsigma, dw_dw0
    return w, dw_dsigma
