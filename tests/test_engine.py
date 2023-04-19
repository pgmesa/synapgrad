import torch
from synapgrad import Tensor
from utils import check_tensors


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
    
    out1 = Tensor.stack((a.squeeze(), b.squeeze()))[0]
    out2 = Tensor.concat((a*c, b), dim=1).transpose(0, 1)[0, :]
    out = out1 @ out2.view(3).unsqueeze(1)
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
    out_t = out1_t @ out2_t.view(3).unsqueeze(1)
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
    c = (a.exp()+b)*b.log().sqrt()
    c.sum().backward()
    
    # torch
    a_t = torch.tensor(l1, requires_grad=True)
    b_t = torch.tensor(l2, requires_grad=True)
    c_t = (a_t.exp()+b_t)*b_t.log().sqrt()
    c_t.sum().backward()
    
    assert check_tensors(c, c_t)
    assert check_tensors(b.grad, b_t.grad)
    
    
def test_engine_v3():
    # Testear unfold max() y min()
    ...