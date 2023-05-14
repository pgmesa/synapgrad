import torch
import synapgrad
from synapgrad import Tensor
from utils import check_tensors
import numpy as np


# Check with pytorch that gradients are correct when applying different tensor operations
def test_engine():
    l1 = [[-4.0, 0.7, 5.0], [6.3, 3.2, 1.3]]
    l2 = [[2.0, 2,  3.0], [2.4, 1.7, 0.5]]
    
    # synapgrad
    a = Tensor(l1, requires_grad=True).unsqueeze(0)**2
    a.retain_grad()
    b = 2**Tensor(l2, requires_grad=True).unsqueeze(0)
    b.retain_grad()
    c = Tensor(4.0, requires_grad=True)
    
    out1 = synapgrad.stack((a.squeeze(), b.squeeze()))[0]
    out2 = synapgrad.concat((a*c, b), dim=1).transpose(0, 1)[0, :]
    out = out1 @ out2.reshape(3).unsqueeze(1)
    s = out.sum()
    s.backward()
    
    ## torch
    a_t = torch.tensor(l1, requires_grad=True).unsqueeze(0)**2
    a_t.retain_grad()
    b_t = 2**torch.tensor(l2, requires_grad=True).unsqueeze(0)
    b_t.retain_grad()
    c_t = torch.tensor(4.0, requires_grad=True)
    
    out1_t = torch.stack((a_t.squeeze(), b_t.squeeze()))[0]
    out2_t = torch.concat((a_t*c_t, b_t), dim=1).transpose(0, 1)[0, :]
    out_t = out1_t @ out2_t.reshape(3).unsqueeze(1)
    s_t = out_t.sum()
    s_t.backward()

    assert check_tensors(a, a_t)
    assert check_tensors(b, b_t)
    assert check_tensors(c, c_t)
    assert check_tensors(out, out_t)
    assert check_tensors(a.grad, a_t.grad)
    assert check_tensors(b.grad, b_t.grad)
    assert check_tensors(c.grad, c_t.grad)


def test_engine_v2():
    l1 = [[[2.0, 4.0], [2.0,4.3]], [[2.0, 4.0], [2.0,4.3]]]
    l2 = [2.0, 4.0]
    
    # synapgrad
    a = Tensor(l1, requires_grad=True)
    b = Tensor(l2, requires_grad=True)
    c = (a.exp()+b)*b.log().sqrt().mean(dim=0, keepdims=True)
    c.sum().backward()
    
    # torch
    a_t = torch.tensor(l1, requires_grad=True)
    b_t = torch.tensor(l2, requires_grad=True)
    c_t = (a_t.exp()+b_t)*b_t.log().sqrt().mean(dim=0, keepdims=True)
    c_t.sum().backward()
    
    assert check_tensors(c, c_t)
    assert check_tensors(b.grad, b_t.grad)
    
    
def test_engine_v3():
    l1 = np.random.rand(4,64,16,16).astype(np.float32)
    print(l1, l1.mean(axis=(0,2,3)))
    
    # synapgrad
    a = Tensor(l1, requires_grad=True)
    c = a.mean(dim=(0,2,3), keepdims=True).max()
    c.sum().backward()
    
    # torch
    a_t = torch.tensor(l1, requires_grad=True)
    c_t = a_t.mean(dim=(0,2,3), keepdims=True).max()
    c_t.sum().backward()
    
    print(a.grad)
    print(a_t.grad)
    
    assert check_tensors(c, c_t)
    assert check_tensors(a.grad, a_t.grad)
    

def test_engine_batchnorm():
    for _ in range(10):
        l1 = np.random.rand(5,64,28,28).astype(np.float32)*10000
        print(np.min(l1), np.max(l1), np.mean(l1, axis=(0,2,3)), np.var(l1, axis=(0,2,3)))
        
        eps = 1e-5

        # synapgrad
        x = Tensor(l1.copy(), requires_grad=True)
        mu = x.mean(dim=(0,2,3), keepdims=True)
        var = ((x - mu)**2).mean(dim=(0,2,3), keepdims=True)
        std = (var + eps).sqrt()
        x_norm = (x - mu) / std 
        x_norm.sum().backward()
        
        # torch
        x_t = torch.tensor(l1.copy(), requires_grad=True)
        mu_t = x_t.mean(dim=(0,2,3), keepdims=True)
        var_t = ((x_t - mu_t)**2).mean(dim=(0,2,3), keepdims=True)
        std_t = (var_t + eps).sqrt()
        x_norm_t = (x_t - mu_t) / std_t
        x_norm_t.sum().backward()
        
        assert check_tensors(mu, mu_t)
        assert check_tensors(var, var_t)
        assert check_tensors(std, std_t)
        assert check_tensors(x_norm, x_norm_t, atol=eps)
        assert check_tensors(x.grad, x_t.grad, atol=eps)
    

def test_engine_unfold():
    x = np.random.randint(0, 3, size=(3,5,8,2)).astype(np.float32)

    dimension = 2
    size = 3
    step = 2

    inp = Tensor(x, requires_grad=True)
    unf = inp.unfold(dimension,size,step)*2
    print(unf)

    inp_t = torch.tensor(x, requires_grad=True)
    unf_t = inp_t.unfold(dimension,size,step)*2
    print(unf_t)

    unf.sum().backward()
    print("Gradient\n", inp.grad)

    unf_t.sum().backward()
    print("Gradient\n", inp_t.grad)
    
    assert check_tensors(unf, unf_t)
    assert check_tensors(inp.grad, inp_t.grad)
    
    
def test_engine_maxpool2d():
    """ 
    Reference:
        https://github.com/pytorch/pytorch/pull/1523#issue-119774673
    """
    l = np.random.randint(0, 10, size=(1,2,4,6)).astype(np.float32)
    print(l)

    kernel_size = (2,2)
    stride = (2,2)

    inp_t = torch.tensor(l, requires_grad=True)
    out_t = inp_t.unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1])
    out_t = out_t.reshape((*out_t.size()[:-2], -1))
    out_t, indices_t = out_t.max(dim=4)
    out_t.backward(torch.ones_like(out_t))

    inp = synapgrad.tensor(l, requires_grad=True)
    out = inp.unfold(2, 2, 2).unfold(3, 2, 2)
    out = out.reshape((*out.shape[:-2], -1))
    out = out.max(dim=4)
    out.backward(synapgrad.ones_like(out))

    print(out); print(out_t)
    print(inp.grad); print(inp_t.grad)
    
    assert check_tensors(out, out_t)
    assert check_tensors(inp.grad, inp_t.grad)
    
    
def test_engine_minpool2d():
    l = np.random.randint(0, 10, size=(1,2,4,6)).astype(np.float32)
    print(l)

    kernel_size = (2,2)
    stride = (2,2)

    inp_t = torch.tensor(l, requires_grad=True)
    out_t = inp_t.unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1])
    out_t = out_t.reshape((*out_t.size()[:-2], -1))
    out_t, indices_t = out_t.min(dim=4)
    out_t.backward(torch.ones_like(out_t))

    inp = Tensor(l, requires_grad=True)
    out = inp.unfold(2, 2, 2).unfold(3, 2, 2)
    out = out.reshape((*out.shape[:-2], -1))
    out = out.min(dim=4)
    out.backward(synapgrad.ones_like(out))

    print(out); print(out_t)
    print(inp.grad); print(inp_t.grad)
    
    assert check_tensors(out, out_t)
    assert check_tensors(inp.grad, inp_t.grad)
    
    
def test_engine_tensor_manipulation():
    # Test Concat, Stack and Unbind
    ...