import torch

t = torch.rand(5)
qt = torch.quantize_per_tensor(t, .1, 0, torch.quint8)

print(t)
print(qt.int_repr())
print(qt.dequantize())
