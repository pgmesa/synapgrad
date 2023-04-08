import torch
import numpy as np
from synapgrad import Tensor


tolerance = 1e-5

def check_tensors(t1:Tensor, t2:torch.Tensor) -> bool:
    """Returns if 2 tensors have the same values and shape

    Args:
        t1 (Tensor): Tensor1
        t2 (Tensor): Tensor2
    """
    if t1 is None or t2 is None: 
        return t1 is t2
    shape_cond = t1.matches_shape(t2)
    values_cond = abs(t1.detach().sum().item() - t2.detach().sum().item()) < abs(tolerance*(np.max(t1.data)+1))
    
    return shape_cond and values_cond