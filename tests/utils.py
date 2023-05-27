from time import time

import torch
import numpy as np
from synapgrad import Tensor
import warnings


def time_fun(function, *args, **kwargs):
    t0 = time()
    out = function(*args, **kwargs)
    tf = time()
    
    return out, tf - t0 

def check_tensors(t1:'Tensor | np.ndarray', t2:torch.Tensor, atol=1e-5, rtol=1e-4, as_np_array=False) -> bool:
    """Returns if 2 tensors have the same values and shape

    Args:
        t1 (Tensor): Tensor1
        t2 (Tensor): Tensor2
    """
    if t1 is None or t2 is None: 
        return t1 is t2
    
    if isinstance(t1, Tensor):
        t1 = t1.data
    elif not as_np_array:
        warnings.warn(f"\nt1 was not type synapgrad.Tensor", stacklevel=0) 
    t1_t = torch.from_numpy(t1)
    
    if t1_t.dtype != t2.dtype:
        if not (t1_t.dtype.is_floating_point and t2.dtype.is_floating_point):
            warnings.warn(f"\ndifferent floating types t1={t1.dtype} t2={t2.dtype}", stacklevel=0) 
        else:
            warnings.warn(f"\ndtype of tensors don't match t1={t1.dtype} t2={t2.dtype}", stacklevel=0) 
        t1_t = t1_t.type(t2.dtype)
    
    check = torch.equal(t1_t, t2) or torch.allclose(t1_t, t2, rtol=rtol, atol=atol)
    
    if not check:
        print(f"Shapes {t1_t.shape} {t2.shape}")
        print("Max Abs error:", (t1_t - t2).abs().max().item())
        print("Max Rel error:", (t1_t - t2).abs().max().item() / (t2.abs().max().item()))
    
    return check