import torch

print(torch.Tensor([2, 4]))
print(torch.eye(3, 3))

print(torch.cuda.is_available())

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
print(y, y.grad)

z = y * y * 3
out = z.mean()

print(z, out, z.grad, out.grad)

out.backward()
print(x.grad, y.grad)