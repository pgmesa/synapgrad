import torch
import synapgrad
from synapgrad.device import Device
import numpy as np
from utils import check_tensors, time_fun

atol = 1e-8; rtol = 1e-5

def op_tester(inputs:list, function, name, device=Device.CPU, module_func=False, nn_functional=False, factor=1, offset=0, backward=True):    
    torch_inputs = [torch.tensor(np.random.rand(*shape)*factor+offset, requires_grad=True, dtype=torch.float32, device=device.value) for shape in inputs]
    if module_func: 
        if nn_functional:
            torch_inputs.insert(0, torch.nn.functional)
        else:
            torch_inputs.insert(0, torch)
    torch_out, torch_fw_time = time_fun(function, *torch_inputs)
    if not isinstance(torch_out, torch.Tensor):
        torch_out = torch_out[0]
    if backward:
        _, torch_bw_time = time_fun(torch_out.backward, torch.ones_like(torch_out))
    
    torch_inputs = torch_inputs[1:] if module_func else torch_inputs
    
    syn_inputs = [synapgrad.tensor(inp.detach().numpy(), requires_grad=True, dtype=np.float32, device=device) for inp in torch_inputs]
    if module_func: 
        if nn_functional:
            syn_inputs.insert(0, synapgrad.nn.functional)
        else:
            syn_inputs.insert(0, synapgrad)
    syn_out, syn_fw_time = time_fun(function, *syn_inputs)
    if not isinstance(syn_out, synapgrad.Tensor):
        syn_out = syn_out[0]
    if backward:
        _, syn_bw_time = time_fun(syn_out.backward, synapgrad.ones_like(syn_out))
    
    syn_inputs = syn_inputs[1:] if module_func else syn_inputs 
    
    if backward:
        print(f'\n{name},  device: {device},  torch/synapgrad ' + 
                f'fp: {torch_fw_time*1000:.2f} / {syn_fw_time*1000:.2f} ms, ' + 
                f'bp: {torch_bw_time*1000:.2f} / {syn_bw_time*1000:.2f} ms')
    else:
        print(f'\n{name},  device: {device},  torch/synapgrad ' + 
                f'fp: {torch_fw_time*1000:.2f} / {syn_fw_time*1000:.2f} ms')
    
    assert check_tensors(syn_out, torch_out, atol=atol, rtol=rtol)
    if backward:
        for synap_inp, torch_inp in zip(syn_inputs, torch_inputs):
            assert check_tensors(synap_inp, torch_inp, atol=atol, rtol=rtol)
            assert check_tensors(synap_inp.grad, torch_inp.grad, atol=atol, rtol=rtol)

# *************************
# ******* Basic ops *******
# *************************

def test_add():
    op_tester([(1000, 1500), (1000, 1500)], lambda x,y: x+y, name='add')

def test_sub():
    op_tester([(1000, 1500), (1000, 1500)], lambda x,y: x-y, name='sub')

def test_mul():
    op_tester([(2000, 2000), (2000, 2000)], lambda x,y: x*y, name='mul')

def test_div():
    op_tester([(500, 800), (500, 800)], lambda x,y: x/y, name='div')

def test_matmul():
    op_tester([(3, 512, 512), (3, 512, 512)], lambda x,y: x@y, name='matmul')
    
def test_addmm():
    op_tester([(10, 10), (10, 10), (10, 10)], lambda engine, x, y, z: engine.addmm(x, y, z), name='addmm', module_func=True)

def test_rsub():
    op_tester([(500, 400, 3)], lambda x: 234 - x, name='rsub')

def test_rmul():
    op_tester([(500, 400, 3)], lambda x: 6.4 * x, name='rmul')

def test_radd():
    op_tester([(500, 400, 3)], lambda x: 34 + x, name='radd')

def test_add_broadcast():
    op_tester([(1000, 1000), (1000, 1)], lambda x,y: x+y, name='add_broadcast')

def test_sub_broadcast():
    op_tester([(1000, 1000), (1000, 1)], lambda x,y: x-y, name='sub_broadcast')

def test_mul_broadcast():
    op_tester([(1000, 1000), (1000, 1)], lambda x,y: x*y, name='mul_broadcast')

def test_div_broadcast():
    op_tester([(1000, 1000), (1000, 1)], lambda x,y: x/y, name='div_broadcast')

def test_neg():
    op_tester([(1000, 1500)], lambda x: -x, name='neg')

def test_pow():
    op_tester([(1000, 1000)], lambda x: x ** 1.1, name='pow')
    op_tester([(1000, 1000)], lambda x: x ** -2.2, name='pow')
    op_tester([(1000, 1000)], lambda x: x ** 0.7, name='pow')
    
def test_slice():
    op_tester([(30, 40, 20, 10)], lambda x: x[10:14, 3:, :, :], name='slice')
    op_tester([(30, 40, 20, 10)], lambda x: x[10:25, :5, :12, 4:], name='slice')
    op_tester([(30, 40, 20, 10)], lambda x: x[10:11, 4:8, :12, 0:-1], name='slice')
    
# *************************
# ******* Other ops *******
# *************************

def test_clone():
    op_tester([(1000, 1000)], lambda x: x.clone(), name='clone')

def test_log():
    op_tester([(1000, 1000)], lambda x: x.log(), name='log', factor=5, offset=0.1)

def test_exp():
    op_tester([(1000, 1000)], lambda x: x.exp(), name='exp')

def test_sqrt():
    op_tester([(1000, 1000)], lambda x: x.sqrt(), name='sqrt')

def test_sum():
    op_tester([(1000, 1500)], lambda x: x.sum(), name='sum')
    op_tester([(1000, 1500)], lambda x: x.sum(dim=1), name='sum')
    op_tester([(1000, 1500, 3)], lambda x: x.sum(dim=(1, 2)), name='sum')
    op_tester([(1000, 1500, 3)], lambda x: x.sum(dim=-1), name='sum')

def test_mean():
    op_tester([(1000, 1500)], lambda x: x.mean(dim=0), name='mean')
    op_tester([(1000, 1500)], lambda x: x.mean(dim=1), name='mean')
    op_tester([(1000, 1500)], lambda x: x.mean(dim=-1), name='mean')

def test_min():
    op_tester([(100, 150)], lambda x: x.min(), name='min')
    op_tester([(1000, 1500)], lambda x: x.min(dim=0), name='min_d0')
    op_tester([(1000, 1500)], lambda x: x.min(dim=1), name='min_d1')
    op_tester([(1000, 1500)], lambda x: x.min(dim=1, keepdims=True), name='min_d1')
    op_tester([(1000, 3, 4, 5)], lambda x: x.min(dim=2), name='min_d2')
    op_tester([(1000, 3, 4, 5)], lambda x: x.min(dim=3), name='min_d3')
    op_tester([(1000, 3, 4, 5)], lambda x: x.min(dim=-1), name='min_d3')
    op_tester([(1000, 3, 4, 5)], lambda x: x.min(dim=-2), name='min_d3')

def test_max():
    op_tester([(100, 150)], lambda x: x.max(), name='max')
    op_tester([(1000, 1500)], lambda x: x.max(dim=0), name='max_d0')
    op_tester([(1000, 1500)], lambda x: x.max(dim=1), name='max_d1')
    op_tester([(1000, 1500)], lambda x: x.max(dim=1, keepdims=True), name='max_d1')
    op_tester([(1000, 3, 4, 5)], lambda x: x.max(dim=2), name='max_d2')
    op_tester([(1000, 3, 4, 5)], lambda x: x.max(dim=3), name='max_d3')
    op_tester([(1000, 3, 4, 5)], lambda x: x.max(dim=-1), name='max_d3')
    op_tester([(1000, 3, 4, 5)], lambda x: x.max(dim=-2), name='max_d3')

def test_squeeze():
    op_tester([(100, 1)], lambda x: x.squeeze(dim=1), name='squeeze')
    op_tester([(1, 3)], lambda x: x.squeeze(dim=0), name='squeeze')
    
def test_unsqueeze():
    op_tester([(1000,)], lambda x: x.unsqueeze(dim=1), name='unsqueeze')

def test_reshape():
    op_tester([(100000,)], lambda x: x.reshape((1000, 100)), name='reshape')
    op_tester([(10000, 200)], lambda x: x.reshape((-1, 50)), name='reshape')
    
def test_movedim():
    op_tester([(100, 200, 300)], lambda x: x.movedim(0, 1), name='movedim')
    op_tester([(10000, 200)], lambda x: x.movedim(-1, -2), name='movedim')
    
def test_flatten():
    op_tester([(100, 200, 300)], lambda x: x.flatten(), name='flatten')
    op_tester([(100, 200, 300)], lambda x: x.flatten(start_dim=0, end_dim=1), name='flatten_0_1')
    op_tester([(100, 200, 300)], lambda x: x.flatten(start_dim=1, end_dim=2), name='flatten_1_2')
    op_tester([(100, 200, 300)], lambda x: x.flatten(start_dim=0, end_dim=2), name='flatten_0_2')

def test_transpose():
    op_tester([(100, 200, 300)], lambda x: x.transpose(2, 1), name='transpose')
    op_tester([(100, 200, 300)], lambda x: x.transpose(0, 1), name='transpose')
    
def test_unfold_dim():
    op_tester([(100, 200, 300)], lambda x: x.unfold(dimension=0, size=2, step=1), name='unfold_dim')
    op_tester([(100, 200, 300)], lambda x: x.unfold(dimension=1, size=2, step=1), name='unfold_dim')
    op_tester([(100, 200, 300)], lambda x: x.unfold(dimension=-1, size=2, step=1), name='unfold_dim')

# **********************************
# ******* Array manipulation *******
# **********************************

def test_stack():
    op_tester([(100,), (100,), (100,)], lambda engine, *x: engine.stack(x, dim=0), name='stack', module_func=True)
    op_tester([(100, 100), (100, 100), (100, 100)], lambda engine, *x: engine.stack(x, dim=1), name='stack', module_func=True)
    op_tester([(100, 100, 100), (100, 100, 100), (100, 100, 100)], lambda engine, *x: engine.stack(x, dim=2), name='stack',module_func=True)

def test_concat():
    op_tester([(100, 200, 4), (100, 200, 3)], lambda engine, *x: engine.concat(x, dim=-1), name='concat', module_func=True)
    op_tester([(100, 200, 4), (22, 200, 4)], lambda engine, *x: engine.concat(x, dim=0), name='concat', module_func=True)

def test_unbind():
    op_tester([(100, 200, 300)], lambda engine, x: engine.unbind(x, dim=-1), name='unbind', module_func=True)
    op_tester([(100, 200, 300)], lambda engine, x: engine.unbind(x, dim=1), name='unbind', module_func=True)

# **************************
# ******* Linear ops *******
# **************************

def test_linear():
    op_tester([(100, 200), (300, 200)], lambda F, x, w: F.linear(x, w), name='linear', module_func=True, nn_functional=True)
    op_tester([(100, 200), (300, 200)], lambda F, x, w: F.linear(x, w), name='linear', module_func=True, nn_functional=True)
    op_tester([(100, 200), (300, 200), (100, 300)], lambda F, x, w, b: F.linear(x, w, b), name='linear', module_func=True, nn_functional=True)

# ************************
# ******* Pool ops *******
# ************************

def test_max_pool1d():
    op_tester([(32, 3, 64)], lambda engine, x: engine.max_pool1d(x, kernel_size=2, stride=1), name='max_pool1d', module_func=True)
    op_tester([(32, 3, 64)], lambda engine, x: engine.max_pool1d(x, kernel_size=3, stride=2), name='max_pool1d', module_func=True)
    op_tester([(32, 3, 64)], lambda engine, x: engine.max_pool1d(x, kernel_size=5, stride=3, padding=2), name='max_pool1d', module_func=True)    

def test_max_pool2d():
    op_tester([(32, 3, 64, 64)], lambda engine, x: engine.max_pool2d(x, kernel_size=(2,2), stride=(1,1)), name='max_pool2d', module_func=True)
    op_tester([(32, 3, 64, 64)], lambda engine, x: engine.max_pool2d(x, kernel_size=(3,2), stride=(1,2)), name='max_pool2d', module_func=True)
    op_tester([(32, 3, 64, 64)], lambda engine, x: engine.max_pool2d(x, kernel_size=(5,3), stride=(3,3), padding=(2,1)), name='max_pool2d', module_func=True)

def test_avg_pool1d():
    op_tester([(32, 3, 64)], lambda engine, x: engine.avg_pool1d(x, kernel_size=2, stride=1), name='avg_pool1d', module_func=True)
    op_tester([(32, 3, 64)], lambda engine, x: engine.avg_pool1d(x, kernel_size=3, stride=2), name='avg_pool1d', module_func=True)
    op_tester([(32, 3, 64)], lambda engine, x: engine.avg_pool1d(x, kernel_size=5, stride=3, padding=2), name='avg_pool1d', module_func=True)

def test_avg_pool2d():
    op_tester([(32, 3, 64, 64)], lambda F, x: F.avg_pool2d(x, kernel_size=(2,2), stride=(1,1)), name='avg_pool2d', module_func=True, nn_functional=True)
    op_tester([(32, 3, 64, 64)], lambda F, x: F.avg_pool2d(x, kernel_size=(3,2), stride=(1,2)), name='avg_pool2d', module_func=True, nn_functional=True)
    op_tester([(32, 3, 64, 64)], lambda F, x: F.avg_pool2d(x, kernel_size=(5,3), stride=(3,3), padding=(2,1)), name='avg_pool2d', module_func=True, nn_functional=True)
    
# ************************
# ******* Conv ops *******
# ************************

def test_unfold():
    op_tester([(32, 3, 64, 64)], lambda F, x: F.unfold(x, kernel_size=(3,3), dilation=1, stride=(1,1), padding=0), name='unfold', module_func=True, nn_functional=True)
    op_tester([(32, 3, 64, 64)], lambda F, x: F.unfold(x, kernel_size=(5,3), dilation=(1,2), stride=(3,2), padding=1), name='unfold', module_func=True, nn_functional=True)
    op_tester([(32, 3, 64, 64)], lambda F, x: F.unfold(x, kernel_size=(7,7), dilation=3, stride=(2,2), padding=(2, 3)), name='unfold', module_func=True, nn_functional=True)
    
def test_fold():
    op_tester([(32, 27, 3844)], lambda F, x: F.fold(x, (64,64), kernel_size=(3,3), dilation=1, stride=(1,1), padding=0), name='fold', module_func=True, nn_functional=True)
    op_tester([(32, 45, 651)], lambda F, x: F.fold(x, (64,64), kernel_size=(5,3), dilation=(1,2), stride=(3,2), padding=1), name='fold', module_func=True, nn_functional=True)
    op_tester([(32, 147, 650)], lambda F, x: F.fold(x, (64,64), kernel_size=(7,7), dilation=3, stride=(2,2), padding=(2, 3)), name='fold', module_func=True, nn_functional=True)

def test_conv1d():
    op_tester([(32, 3, 64)], lambda engine, x: engine.conv1d(x, weight=engine.ones((34, 3, 3), requires_grad=True), bias=engine.zeros(34, requires_grad=True), stride=1, padding=0, dilation=2), name='conv1d', module_func=True)
    op_tester([(32, 3, 64)], lambda engine, x: engine.conv1d(x, weight=engine.ones((34, 3, 5), requires_grad=True), bias=None, stride=2, padding=1), name='conv1d', module_func=True)
    op_tester([(32, 3, 64)], lambda engine, x: engine.conv1d(x, weight=engine.ones((34, 3, 7), requires_grad=True), bias=engine.zeros(34, requires_grad=True), stride=2, padding=3), name='conv1d', module_func=True)
    
def test_conv2d():
    op_tester([(32, 3, 64, 64)], lambda engine, x: engine.conv2d(x, weight=engine.ones((34, 3, 3, 3), requires_grad=True), bias=engine.zeros(34, requires_grad=True), stride=(1,1), padding=0, dilation=2), name='conv2d', module_func=True)
    op_tester([(32, 3, 64, 64)], lambda engine, x: engine.conv2d(x, weight=engine.ones((34, 3, 5, 3), requires_grad=True), bias=None, stride=(1,2), padding=1), name='conv2d', module_func=True)
    op_tester([(32, 3, 64, 64)], lambda engine, x: engine.conv2d(x, weight=engine.ones((34, 3, 7, 7), requires_grad=True), bias=engine.zeros(34, requires_grad=True), stride=(2,2), padding=(2, 3)), name='conv2d', module_func=True)