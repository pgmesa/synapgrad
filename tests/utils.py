import torch
import numpy as np
from synapgrad import Tensor
import warnings


def check_tensors(t1:'Tensor | np.ndarray', t2:torch.Tensor, atol=1e-8, rtol=0.00001) -> bool:
    """Returns if 2 tensors have the same values and shape

    Args:
        t1 (Tensor): Tensor1
        t2 (Tensor): Tensor2
    """
    if t1 is None or t2 is None: 
        return t1 is t2
    
    if isinstance(t1, Tensor):
        t1 = t1.data
    t1_t = torch.from_numpy(t1)
    
    if t1_t.dtype != t2.dtype:
        if not (t1_t.dtype.is_floating_point and t2.dtype.is_floating_point):
            warnings.warn(f"\ndifferent floating types t1={t1.dtype} t2={t2.dtype}", stacklevel=0) 
        else:
            warnings.warn(f"\ndtype of tensors don't match t1={t1.dtype} t2={t2.dtype}", stacklevel=0) 
        t1_t = t1_t.type(t2.dtype)
    
    return torch.equal(t1_t, t2) or torch.allclose(t1_t, t2, rtol=rtol, atol=atol)