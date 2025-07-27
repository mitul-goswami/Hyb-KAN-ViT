import torch
import numpy as np

def bspline_basis(x, knots, degree, derivative=0):
    n = len(knots) - degree - 1
    basis = torch.zeros(len(x), n, device=x.device)
    for i in range(n):
        basis[:, i] = ((knots[i] <= x) & (x < knots[i+1])).float()
    for d in range(1, degree+1):
        new_basis = torch.zeros_like(basis)
        for i in range(n - d):
            left = (x - knots[i]) / (knots[i+d] - knots[i] + 1e-8)
            right = (knots[i+d+1] - x) / (knots[i+d+1] - knots[i+1] + 1e-8)
            term1 = left * basis[:, i]
            term2 = right * basis[:, i+1]
            new_basis[:, i] = term1 + term2
        basis = new_basis
    if derivative > 0:
        for _ in range(derivative):
            deriv_basis = torch.zeros_like(basis)
            for i in range(n-1):
                left = degree / (knots[i+degree] - knots[i] + 1e-8)
                right = degree / (knots[i+degree+1] - knots[i+1] + 1e-8)
                deriv_basis[:, i] = left * basis[:, i] - right * basis[:, i+1]
            basis = deriv_basis
    return basis
