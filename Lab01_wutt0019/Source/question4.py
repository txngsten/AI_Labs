# Student Name: Oliver Wuttke
# Student FAN: WUTT0019
# File: question4.py
# Date: 31-07-2026
# Description: Performing matrix multiplication between two tensors

import torch

x = torch.rand(2, 3)
y = torch.rand(3, 2)
z = torch.matmul(x, y)

print("Matrix multiplication between two tensors 2x3 and 3x2:", z)