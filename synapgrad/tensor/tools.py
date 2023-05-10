from typing import Iterable
import numpy as np
from .tensor import Tensor


# ****************************
# ******* Initializers *******
# ****************************

def tensor(data, requires_grad=False, dtype=None) -> 'Tensor':
    """
    Creates a Tensor from a numpy array
    """
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)

def ones(shape, dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with ones
    """
    return Tensor(np.ones(shape), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def ones_like(tensor:'Tensor', dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with ones
    """
    return Tensor(np.ones_like(tensor.data), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def zeros(shape, dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with zeros
    """
    return Tensor(np.zeros(shape), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def zeros_like(tensor:'Tensor', dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with zeros
    """
    return Tensor(np.zeros_like(tensor.data), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def arange(*interval, dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with values in range
    """
    return Tensor(np.arange(*interval), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def randn(*shape, dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with values drawn from the standard Gaussian distribution
    """
    return Tensor(np.random.randn(*shape), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def normal(loc, scale, *shape, dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with values drawn from a custom Gaussian distribution
    """
    return Tensor(np.random.normal(loc, scale, *shape), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def randint(low, high, *shape,  dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with integer values drawn in the range between low and high
    """
    return Tensor(np.random.randint(low, high, *shape), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def eye(dim, dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a 2-dimensional Tensor (matrix) equal to the identity matrix.
    """
    return Tensor(np.eye(dim), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

# ***********************************
# ******* Tensor manipulation *******
# ***********************************

def concat(tensors:Iterable['Tensor'], dim=0) -> 'Tensor':
    r_grad = False
    for t in tensors:
        if not isinstance(t, Tensor):
            raise ValueError("All elements must be Tensors")
        r_grad = r_grad or t.requires_grad
    
    # Check that all tensors have the same shape along the specified dim
    dim_sizes = [tensor.shape[dim] for tensor in tensors]
    assert all(size == dim_sizes[0] for size in dim_sizes), f"Shapes along dim {dim} don't match: {[tensor.shape for tensor in tensors]}"

    # Concatenate the sections along the specified dim
    new_data = np.concatenate([tensor.data for tensor in tensors], axis=dim)

    out = Tensor(new_data, tensors, _operation='<Concat>', requires_grad=r_grad)

    def _backward():
        # Split the gradient along the concatenated dim and backpropagate to each input tensor
        grads = np.split(out._grad, len(tensors), axis=dim)
        for tensor, grad in zip(tensors, grads):
            if not tensor.requires_grad: continue
            tensor._grad += grad

    out._backward = _backward
    
    return out


def stack(tensors:Iterable['Tensor'], dim=0) -> 'Tensor':
    r_grad = False
    for t in tensors:
        if not isinstance(t, Tensor):
            raise ValueError("All elements must be Tensors")
        r_grad = r_grad or t.requires_grad

    # Stack data along the specified dim
    new_data = np.stack([tensor.data for tensor in tensors], axis=dim)

    out = Tensor(new_data, tensors, _operation='<Stack>', requires_grad=r_grad)

    def _backward():
        # Split the gradient along the concatenated dim and backpropagate to each input tensor
        grads = np.rollaxis(out._grad, axis=dim)
        for tensor, grad in zip(tensors, grads):
            if not tensor.requires_grad: continue
            tensor._grad += grad

    out._backward = _backward
    
    return out