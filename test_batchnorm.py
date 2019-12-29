import timeit
import nnpy
import numpy as np
import torch

b = 128
c = 3
l_in = 64

bnt = torch.nn.BatchNorm1d(l_in, eps=1e-5, momentum=0.5, affine=True)

x = np.random.rand(b, l_in).astype(np.float32)
o = np.random.rand(b, l_in).astype(np.float32)
xt = torch.tensor(x, requires_grad=True)
ot = torch.tensor(o, requires_grad=False)

bn = nnpy.BatchNorm(l_in, eps=1e-5, momentum=0.5)
bn.gamma = bnt.weight.detach().numpy()
bn.beta = bnt.bias.detach().numpy()

bnt.zero_grad()

start = timeit.default_timer()
for i in range(1000):
    rt = bnt.forward(xt)
end = timeit.default_timer()
print('tf:', end-start)

start = timeit.default_timer()
for i in range(1000):
    r = bn.forward(x)
end = timeit.default_timer()
print('nf:', end-start)

print('check_forward:', np.allclose(rt.detach().numpy(), r))
print(np.sum(np.abs(rt.detach().numpy() - r)))

start = timeit.default_timer()
for i in range(1):
    rt.backward(ot)
end = timeit.default_timer()
print('tb:', end-start)

start = timeit.default_timer()
for i in range(1):
    b = bn.backward(o)
end = timeit.default_timer()
print('nb:', end-start)

print('check_backward:', np.allclose(bnt.weight.grad.detach().numpy(), bn.grad_gamma))
print('check_backward:', np.allclose(bnt.bias.grad.detach().numpy(), bn.grad_beta))
print('check_backward:', np.sum(np.abs(bnt.weight.grad.detach().numpy() - bn.grad_gamma)))
