import torch

# unbroken gradient, backward goes all the way to x
x = torch.ones(2, 2, requires_grad=True)
y = 2 * x + 2
z = y * y * 3
out = z.mean()
out.backward()
print(x.grad)
baseline_x = x.grad

# broken gradient, ends at _y
x = torch.ones(2, 2, requires_grad=True)
y = 2 * x + 2

_y = torch.tensor(y.detach(), requires_grad=True)
z = _y * _y * 3
out = z.mean()
out.backward()
print(x.grad)
print(_y.grad)

# we can however, take the grad of _y and put it in manually!
y.backward(_y.grad)
print(x.grad)
x_from_manipulated_graph = x.grad

assert torch.eq(x_from_manipulated_graph, baseline_x).all()