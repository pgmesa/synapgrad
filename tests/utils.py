import torch
import numpy as np
from synapgrad import Tensor


rtol = 0.00001
atol = 1e-8

def check_tensors(t1:Tensor, t2:torch.Tensor) -> bool:
    """Returns if 2 tensors have the same values and shape

    Args:
        t1 (Tensor): Tensor1
        t2 (Tensor): Tensor2
    """
    if t1 is None or t2 is None: 
        return t1 is t2
    
    t1_t = torch.from_numpy(t1.data)
    
    assert t1_t.dtype == t2.dtype, f"dtype of tensors don't match t1={t1.dtype} t2={t2.dtype}"
    
    return torch.equal(t1_t, t2) or torch.allclose(t1_t, t2, rtol=rtol, atol=atol)