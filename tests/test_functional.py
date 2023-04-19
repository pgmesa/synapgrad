import torch
from synapgrad import Tensor, nn
from utils import check_tensors
import numpy as np


def test_fold_unfold():
    l = np.random.randn(2,3,5,4).astype(np.float32)
    print(l)
    
    kernel_size = (3,2); stride = (2,3); padding = (2,3); dilation = (2,3)
    
    inp = torch.tensor(l, dtype=torch.float)
    # unfolde
    unf_t = torch.nn.functional.unfold(inp, kernel_size, stride=stride, padding=padding, dilation=dilation)
    unf = nn.functional.unfold(l, kernel_size, stride=stride, padding=padding, dilation=dilation)
    
    print(unf_t)
    print(unf)

    # fold
    f_t = torch.nn.functional.fold(unf_t, l.shape[2:], kernel_size, stride=stride, padding=padding, dilation=dilation)
    f = nn.functional.fold(unf, l.shape[2:], kernel_size, stride=stride, padding=padding, dilation=dilation)
    
    print(f_t)
    print(f)
    
    check_tensors(unf, unf_t)
    check_tensors(f, f_t)
