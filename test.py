from deepqnetwork import MLP
import torch

x = torch.randn((128), requires_grad=True)
# c = torch.randn((128), requires_grad = True)
# x = c * 8
# print(x.grad)
test_model = MLP(128, 8)
x = test_model(x)
print(x.grad_fn)

h = [1, 2, 3, 4, 5]
print(h[0, 2, 3])