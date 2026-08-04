# Student Name: Oliver Wuttke
# Student FAN: WUTT0019
# File: question2.py
# Date: 31-07-2026
# Description: Creating a 3x3 2D tensor (Matrix) with random values and printing the matrix and the shape

import torch

x = torch.rand(3, 3)
print("3x3 Tensor with random values:", x)
print("With shape:", x.shape)