import torch

arr = torch.ones(4, 5, 5)
arr = arr[:, None, :, :]

arr2 = torch.ones(4, 2, 5, 5)
print(arr.shape)
print((arr - arr2).shape)
