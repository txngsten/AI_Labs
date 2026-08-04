# Student Name: Oliver Wuttke
# Student FAN: WUTT0019
# File: question5.py
# Date: 31-07-2026
# Description: Converting a NumPy array into a PyTorch tensor

import torch
import numpy as np

numpy_array = np.array([1, 2, 3, 4, 5])
tensor_from_np_array = torch.from_numpy(numpy_array)
tensor_addition = tensor_from_np_array + 10

print("Tensor from NumPy array:", tensor_from_np_array)
print("Result of adding 10 to the tensor:", tensor_addition)