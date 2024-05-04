import torch
from torch.utils.data import DataLoader

from rnn import TextDataset, LinearFunction, TanhFunction, ReLUFunction

from torch.autograd import gradcheck

input = torch.randn((2, 3), dtype=torch.double, requires_grad=True)
weight = torch.randn((4, 3), dtype=torch.double, requires_grad=True)
bias = torch.randn(4, dtype=torch.double, requires_grad=True)

# Use gradcheck to test the LinearFunction
test = gradcheck(LinearFunction.apply, (input, weight, bias), eps=1e-6, atol=1e-4)
print("Gradcheck passed:", test)

input_relu = torch.randn((3, 2), dtype=torch.double, requires_grad=True)
input_tanh = torch.randn((3, 2), dtype=torch.double, requires_grad=True)

relu_check = gradcheck(ReLUFunction.apply, input_relu, eps=1e-6, atol=1e-4)
print("ReLUFunction check:", relu_check)

# Verify TanhFunction
tanh_check = gradcheck(TanhFunction.apply, input_tanh, eps=1e-6, atol=1e-4)
print("TanhFunction check:", tanh_check)