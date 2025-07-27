import torch
import torch.nn as nn
import numpy as np

class BSplineActivation(nn.Module):
    def __init__(self, in_features, grid_size=5, spline_order=3, grid_range=[-1.5, 1.5], share_grid_across_features=False):
        super().__init__()
        self.in_features = in_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.share_grid = share_grid_across_features
        if self.share_grid:
            self.knots = nn.Parameter(torch.linspace(grid_range[0], grid_range[1], grid_size))
        else:
            self.knots = nn.Parameter(torch.linspace(grid_range[0], grid_range[1], grid_size).repeat(in_features, 1))
        self.coeffs = nn.Parameter(torch.randn(in_features, grid_size + spline_order - 1))
        self.scaler = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(-1, self.in_features)
        if self.share_grid:
            knots = self.knots
        else:
            knots = self.knots.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        basis = self.bspline_basis(x, knots)
        spline_out = torch.einsum('bi,bij->bj', self.coeffs, basis)
        spline_out = spline_out * self.scaler
        return spline_out.view(batch_size, seq_len, self.in_features)

    def bspline_basis(self, x, knots):
        n_basis = knots.shape[-1] - self.spline_order
        basis = torch.zeros(x.shape[0], x.shape[1], n_basis, device=x.device)
        for i in range(n_basis):
            left = knots[..., i]
            right = knots[..., i + self.spline_order]
            mask = (x >= left) & (x < right)
            basis[..., i] = mask.float()
        for d in range(1, self.spline_order + 1):
            for i in range(n_basis):
                left_term = (x - knots[..., i]) / (knots[..., i + d] - knots[..., i] + 1e-8) * basis[..., i]
                right_term = (knots[..., i + d + 1] - x) / (knots[..., i + d + 1] - knots[..., i + 1] + 1e-8) * basis[..., i + 1]
                basis[..., i] = left_term + right_term
        return basis
