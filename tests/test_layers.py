import torch
from synapgrad import Tensor, nn
from utils import check_tensors
import numpy as np
    
    
def test_linear():
    list_ = [[0.2, 1, 4, 2, -1, 4], [0.2, 1, 4, 2, -1, 4]]
    
    # synapgrad
    inp = Tensor(list_, requires_grad=True)
    linear = nn.Linear(6,3)
    out = linear(inp)
    out = out.sum()
    out.backward()

    # torch
    inp_t = torch.tensor(list_, requires_grad=True)
    linear_t = torch.nn.Linear(6,3)
    out_t = linear_t(inp_t)
    out_t = out_t.sum()
    out_t.backward()

    params = linear.parameters()
    params_t = list(linear_t.parameters())

    assert len(params) == len(params_t)
    for p, p_t in zip(params, params_t):
        assert check_tensors(p.grad, p_t.grad)
        
        
def test_flatten():
    l = np.random.randn(30,28,28,3,4)
    
    # synapgrad
    inp = Tensor(l, requires_grad=True)
    linear = nn.Flatten(start_dim=1, end_dim=2)
    out_l = linear(inp)
    out = out_l.sum()
    out.backward()

    # torch
    inp_t = torch.tensor(l, requires_grad=True)
    linear_t = torch.nn.Flatten(start_dim=1, end_dim=2)
    out_tl = linear_t(inp_t)
    out_t = out_tl.sum()
    out_t.backward()

    assert check_tensors(out_l, out_tl)
    assert check_tensors(inp.grad, inp_t.grad)
    

def test_fold_unfold():
    l = np.random.randn(2,3,5,4).astype(np.float32)
    print(l)
    
    kernel_size = (3,2); stride = (2,3); padding = (2,3); dilation = (2,3)
    
    inp = Tensor(l, requires_grad=True)
    inp_t = torch.tensor(l, requires_grad=True)
    # unfolde
    unf_t = torch.nn.Unfold(kernel_size, stride=stride, padding=padding, dilation=dilation)(inp_t)*3
    unf = nn.Unfold(kernel_size, stride=stride, padding=padding, dilation=dilation)(inp)*3
    
    print(unf_t)
    print(unf)

    # fold
    f_t = torch.nn.Fold(l.shape[2:], kernel_size, stride=stride, padding=padding, dilation=dilation)(unf_t)
    f = nn.Fold(l.shape[2:], kernel_size, stride=stride, padding=padding, dilation=dilation)(unf)
    
    f_t.sum().backward()
    f.sum().backward()
    
    print(f_t)
    print(f)
    
    print(inp.grad)
    print(inp_t.grad)
    
    check_tensors(unf, unf_t)
    check_tensors(f, f_t)
    check_tensors(inp.grad, inp_t.grad)
    
    
def test_maxpool2d():
    l = np.random.randn(32,5,28,28).astype(np.float32)
    l2 = np.random.randn(32,5,1,1).astype(np.float32)
    
    kernel_size = 5; stride = None; padding = 1; dilation = 2
    
    inp = Tensor(l, requires_grad=True)
    p2 = Tensor(l2)
    pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    out = pool(inp)*p2
    out.sum().backward()
    
    inp_t = torch.tensor(l, requires_grad=True)
    p2_t = torch.tensor(l2)
    pool_t = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    out_t = pool_t(inp_t)*p2_t
    out_t.sum().backward()
    
    print(out.shape, "\n" + str(out)[:100])
    print(out_t.shape, "\n" + str(out_t)[:100])
    
    print(inp.grad.shape, "\n" + str(inp.grad)[:200])
    print(inp_t.grad.shape, "\n" + str(inp_t.grad)[:200])
        
    assert check_tensors(out, out_t)
    assert check_tensors(inp.grad, inp_t.grad)
    

def test_conv2d():
    l = np.random.rand(2,3,4,4).astype(np.float32)
    out_channels = 5; kernel_size = 3; stride = 1; padding = 'same'

    inp = Tensor(l, requires_grad=True)
    conv = nn.Conv2d(l.shape[1], out_channels, kernel_size, stride, padding)
    out = conv(inp)
    out.sum().backward()
    
    inp_t = torch.tensor(l, requires_grad=True)
    conv_t = torch.nn.Conv2d(l.shape[1], out_channels, kernel_size, stride, padding)
    out_t = conv_t(inp_t)
    out_t.sum().backward()

    print(out_t.shape)
    print(out.shape)
    
    assert out.matches_shape(out_t.detach().numpy())

    params_t = list(conv_t.parameters())
    params = list(conv.parameters())
    for i, (p_t, p) in enumerate(zip(params_t, params)):
        print(f"Param {i+1}")
        print(p_t); print(p)
        print(p_t.grad); print(p.grad)
        check_tensors(p.grad, p_t.grad)
    

def test_dropout():
    l = np.random.randn(32,5,28,28).astype(np.float32)
    
    inp = Tensor(l, requires_grad=True)
    dropout = nn.Dropout(p=0.3)
    out = dropout(inp)
    out.sum().backward()
    
    dropout.eval()
    out2 = dropout(out)
    
    inp_t = torch.tensor(l, requires_grad=True)
    dropout_t = torch.nn.Dropout(p=0.3)
    out_t = dropout_t(inp_t)
    out_t.sum().backward()
        
    assert out.matches_shape(Tensor(out_t.detach().numpy()))
    assert check_tensors(out, torch.tensor(out2.data))
    
    
def test_batchnorm1d():
    for _ in range(5):
        l = np.random.randn(5,64).astype(np.float32)*10000
        momentum = 0.5; affine = True; track_running_stats = True; eps = 1e-5

        bnorm_t = torch.nn.BatchNorm1d(
            l.shape[1], momentum=momentum, affine=affine,
            track_running_stats=track_running_stats, eps=eps)

        inp_t = torch.tensor(l.copy(), requires_grad=True)
        out_t = bnorm_t(inp_t)
        out_t.sum().backward()

        inp = Tensor(l.copy(), requires_grad=True)
        bnorm = nn.BatchNorm1d(
            inp.shape[1], momentum=momentum, affine=affine,
            track_running_stats=track_running_stats, eps=eps)
        out = bnorm(inp)
        out.sum().backward()

        print(out_t)
        print(out)

        print(inp_t.grad)
        print(inp.grad)
        
        check_tensors(out, out_t, atol=eps, rtol=1e-3)
        check_tensors(inp.grad, inp_t.grad, atol=eps, rtol=1e-3)
        
        if track_running_stats:
            print("TRACKED STATS")
            print(bnorm_t.running_mean)
            print(bnorm.running_mean)
            print(bnorm_t.running_var)
            print(bnorm.running_var)
            print("=============")
            check_tensors(bnorm.running_mean, bnorm_t.running_mean)
            check_tensors(bnorm.running_var, bnorm_t.running_var)

        params_t = list(bnorm_t.parameters())
        params = list(bnorm.parameters())
        for i, (p_t, p) in enumerate(zip(params_t, params)):
            print(f"Param {i+1}")
            print(p_t); print(p)
            print(p_t.grad); print(p.grad)
            check_tensors(p, p_t)
            check_tensors(p.grad, p_t.grad)


def test_batchnorm2d():
    for _ in range(5):
        l = np.random.randn(5,64,28,28).astype(np.float32)*10000
        momentum = 0.5; affine = True; track_running_stats = True; eps = 1e-5

        bnorm_t = torch.nn.BatchNorm2d(
            l.shape[1], momentum=momentum, affine=affine,
            track_running_stats=track_running_stats, eps=eps)

        inp_t = torch.tensor(l.copy(), requires_grad=True)
        out_t = bnorm_t(inp_t)
        out_t.sum().backward()

        inp = Tensor(l.copy(), requires_grad=True)
        bnorm = nn.BatchNorm2d(
            inp.shape[1], momentum=momentum, affine=affine,
            track_running_stats=track_running_stats, eps=eps)
        out = bnorm(inp)
        out.sum().backward()

        print(out_t)
        print(out)

        print(inp_t.grad)
        print(inp.grad)
        
        check_tensors(out, out_t, atol=eps, rtol=1e-3)
        check_tensors(inp.grad, inp_t.grad, atol=eps, rtol=1e-3)
        
        if track_running_stats:
            print("TRACKED STATS")
            print(bnorm_t.running_mean)
            print(bnorm.running_mean)
            print(bnorm_t.running_var)
            print(bnorm.running_var)
            print("=============")
            check_tensors(bnorm.running_mean, bnorm_t.running_mean)
            check_tensors(bnorm.running_var, bnorm_t.running_var)

        params_t = list(bnorm_t.parameters())
        params = list(bnorm.parameters())
        for i, (p_t, p) in enumerate(zip(params_t, params)):
            print(f"Param {i+1}")
            print(p_t); print(p)
            print(p_t.grad); print(p.grad)
            check_tensors(p, p_t)
            check_tensors(p.grad, p_t.grad)