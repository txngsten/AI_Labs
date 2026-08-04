# Student Name: Oliver Wuttke
# Student FAN: WUTT0019
# File: question3.py
# Date: 31-07-2026
# Description: Performing element-wise addition and multiplication between tensors

import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

tensor_add = x + y
tensor_mult = x * y

print("Element-wise addition between two tensors:", tensor_add)
print("Element-wise multiplication between two tensors:", tensor_mult)