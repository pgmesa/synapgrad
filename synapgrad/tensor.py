import importlib
from typing import Union

import numpy as np

from synapgrad.device import Device


default_type__ = np.float32

F = None
autograd = None
tools = None

def lazy_import():
    global F, autograd, tools
    F = importlib.import_module("synapgrad.functional")
    autograd = importlib.import_module("synapgrad.autograd")
    tools = importlib.import_module("synapgrad.tools")

# ****************************
# ******* Initializers *******
# ****************************

def tensor(data, requires_grad=False, dtype=None, device=None) -> 'Tensor':
    """
    Creates a Tensor from a numpy array
    """
    data = np.array(data).astype(default_type__)
    return Tensor(data, requires_grad=requires_grad, dtype=dtype, device=device)

def ones(shape, dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with ones
    """
    return Tensor(np.ones(shape).astype(default_type__), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def ones_like(tensor:'Tensor', dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with ones
    """
    return Tensor(np.ones_like(tensor.data), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def zeros(shape, dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with zeros
    """
    return Tensor(np.zeros(shape).astype(default_type__), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def zeros_like(tensor:'Tensor', dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with zeros
    """
    return Tensor(np.zeros_like(tensor.data), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def arange(*interval, dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with values in range
    """
    return Tensor(np.arange(*interval).astype(default_type__), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def rand(*shape, dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with values drawn from the uniform distribution
    """
    return Tensor(np.random.rand(*shape).astype(default_type__), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def randn(*shape, dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with values drawn from the standard Gaussian distribution
    """
    return Tensor(np.random.randn(*shape).astype(default_type__), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def normal(loc, scale, *shape, dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with values drawn from a custom Gaussian distribution
    """
    return Tensor(np.random.normal(loc, scale, *shape).astype(default_type__), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def randint(low, high, *shape,  dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a Tensor filled with integer values drawn in the range between low and high
    """
    return Tensor(np.random.randint(low, high, *shape).astype(default_type__), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

def eye(dim, dtype=None, requires_grad=False, name=None, device=None):
    """
    Creates a 2-dimensional Tensor (matrix) equal to the identity matrix.
    """
    return Tensor(np.eye(dim).astype(default_type__), dtype=dtype, requires_grad=requires_grad, name=name, device=device)

# ****************************
# ******* Tensor Class *******
# ****************************

class Tensor:
    
    def __init__(self, data, children:tuple=(), operation:str=None, requires_grad:bool=False, dtype=None, name:str=None, device:Device=None) -> None:
        """
        Creates a Tensor object from the given data, which is always transformed internally into a numpy array.

        Args:
            data (number or iterable): data of the tensor, must be convertible into a numpy.array().
            children (tuple, optional): tensors which produced this tensor as a result of an operation. Defaults to ().
            operation (str, optional): string that represents the operation that created this tensor. Defaults to None.
            requires_grad (bool, optional): whether this tensor requieres gradients or not. Defaults to False.
            dtype (type, optional): numpy type of this tensor data. Defaults to None.
            device (Device, optional): device on which this tensor is located. Defaults to None.
            name (str, optional): name of this tensor. Defaults to None.

        Raises:
            RuntimeError: if data is not convertible into a numpy.array()

        Examples:
            >>> Tensor(np.array([1, 2, 3]))
            >>> Tensor(np.array([1, 2, 3]), dtype=np.float32)
            >>> Tensor(5.0, requires_grad=True, children=(a,b), operation="<Add>")
            >>> Tensor(np.array([1, 2, 3]), dtype=np.float32, requires_grad=True, name="weights", device=Device.CPU)
            >>> Tensor([0.0], dtype=np.float64, requires_grad=True, name="bias", device=Device.CPU)
            
        """
        if F is None: lazy_import()
        
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=default_type__)
        if dtype is not None and data.dtype != dtype: data = data.astype(dtype)
        assert isinstance(data, np.ndarray), "data must be a list or numpy array"
        
        self.data = data
        self.device = device if device is not None else Device.CPU
        # Internal variables
        self._grad = None
        self._grad_fn = None
        req_grad = requires_grad and autograd.gradient__
        if req_grad and not self.is_floating_point:
            raise RuntimeError("Only floating point Tensors can require gradients")
        self._requires_grad = req_grad
        self._retain_grad = False
        self._children = children
        self._operation = operation
        self._name = name
        
    # **************************
    # ******* Properties *******
    # ************************** 
    
    @property
    def name(self) -> str:
        return "" if self._name is None else str(self._name)   
    
    @name.setter
    def name(self, name):
        self._name = name
        
    @property
    def shape(self) -> tuple:
        return self.data.shape
    
    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype
    
    @property
    def size(self) -> int:
        return self.data.size
    
    @property
    def ndim(self) -> int:
        return self.data.ndim
    
    @property
    def is_leaf(self) -> bool:
        return not self.requires_grad or self.grad_fn is None
    
    @property
    def is_floating_point(self) -> bool:
        return tools.is_floating_point(self.data)
    
    @property
    def requires_grad(self) -> bool:
        return self._requires_grad
    
    @requires_grad.setter
    def requires_grad(self, value:bool):
        if not self.is_leaf:
            raise RuntimeError("you can only change requires_grad flags of leaf variables. " + 
                    "If you want to use a computed variable in a subgraph that doesn't require " + 
                    "differentiation use var_no_grad = var.detach()")
        
        if value and not self.is_floating_point:
            raise RuntimeError("Only floating point Tensors can require gradients")
        
        self._requires_grad = value
    
    @property
    def grad(self) -> 'Tensor':
        if not self.is_leaf and not self.has_grad() and not self._retain_grad:
            print("\n[!] WARNING: The .grad attribute of a Tensor that is not a " + 
                  "leaf Tensor is being accessed. Its .grad attribute won't be populated " + 
                  "during autograd.backward(). If you indeed want the .grad field to be populated " + 
                  "for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor")
        return Tensor(self._grad, device=self.device) if self._grad is not None else None

    @grad.setter
    def grad(self, grad:'Tensor'):
        if not self.matches_shape(grad):
            raise RuntimeError(f"Attempt to assign grad ({grad.shape}) to  a Tensor ({self.shape}) that has a different shape")
        self._grad = grad.data
    
    @property
    def grad_fn(self) -> 'Tensor':
        return self._grad_fn
    
    @grad_fn.setter
    def grad_fn(self, grad_fn):
        if grad_fn is not None and not self.requires_grad:
            raise RuntimeError("Cannot set grad_fn for a Tensor that doesn't require grad")
        self._grad_fn = grad_fn
    
    # *********************************
    # ******* Utility functions *******
    # *********************************
    
    def numel(self) -> int:
        return self.data.size
    
    def has_grad(self) -> bool:
        return self._grad is not None
    
    def retain_grad(self):
        """ Grad is not stored in not leaf tensors by default to avoid extra memory consumption. Call
        this function to enable grad storing"""
        if not self.requires_grad:
            raise RuntimeError("Cannot retain_grad() on a Tensor that doesn't require grad")
        self._retain_grad = True
            
    def matches_shape(self, tensor:Union['Tensor', np.ndarray]) -> bool:
        if len(self.shape) != len(tensor.shape):
            return False
        
        for n1, n2 in zip(self.shape, tensor.shape):
            if n1 != n2: return False
        
        return True
    
    # *********************************
    # ******* Data manipulation *******
    # *********************************
    
    def numpy(self) -> np.ndarray:
        if self.requires_grad:
            raise RuntimeError("Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead")
        return self.data
    
    def item(self) -> float:
        squeezed = self.data.squeeze()
        if len(squeezed.shape) > 0:
            raise ValueError("only one element tensors can be converted to Python scalars")
        
        return squeezed
        
    def detach(self) -> 'Tensor':
        """
        Detach a tensor from its graph. It returns a copy of the tensor with requires_grad=False.
        This means that the tensor won't be updated when the graph is updated.
        """
        return Tensor(self.data.copy(), requires_grad=False, name=self.name, device=self.device)
    
    def clone(self) -> 'Tensor':
        """ 
        Clones a tensor
        """
        return F.clone(self)
    
    # *********************************
    # *********** Backprop ************
    # *********************************
    
    def backward(self, grad:'Tensor'=None):
        """
        Computes the gradient of current tensor w.r.t. graph leaves.

        The graph is differentiated using the chain rule. If the tensor is non-scalar 
        (i.e. its data has more than one element) and requires gradient, the function 
        additionally requires specifying gradient. It should be a tensor of matching type
        and location, that contains the gradient of the differentiated function w.r.t. self.

        This function accumulates gradients in the leaves - you might need to zero .grad attributes
        or set them to None before calling it.
        """
        if not self.requires_grad:
            raise RuntimeError("Trying to call backward on Tensor with requires_grad=False")

        if grad is None:
            if self.data.size > 1:
                raise RuntimeError("grad must be specified for non-scalar tensors")
            else:
                grad = Tensor(1.0, dtype=self.dtype, device=self.device)
        
        assert tools.is_floating_point(grad), "expected float dtype for grad, got %s" % grad.dtype
          
        if not isinstance(grad, Tensor):
            raise ValueError("Gradient parameter must be a Tensor")
        
        autograd.backward(self.grad_fn, grad)
        

    def zero_(self):
        self.grad = Tensor(np.zeros_like(self.data), device=self.device)
    
    # ****************************************
    # *********** CPU/GPU support ************
    # ****************************************
    
    def __move_data(self, device):
        ...
    
    def to(self, device:Device) -> 'Tensor':
        if self.device != device:
            self.__move_data(device)
        return self
    
    def cpu(self) -> 'Tensor':
        return self.to(Device.CPU)
    
    def gpu(self) -> 'Tensor':
        raise NotImplementedError

    # *************************
    # ******* Basic ops *******
    # *************************
    
    def __add__(self, summand:'Tensor') -> 'Tensor':
        summand = summand if isinstance(summand, Tensor) else Tensor(summand, device=self.device)
        from . import functional as F
        return  F.add(self, summand)
        
        
    def __mul__(self, factor:'Tensor') -> 'Tensor':
        factor = factor if isinstance(factor, Tensor) else Tensor(factor, device=self.device)
        from . import functional as F
        return F.mul(self, factor)
    
    
    def __matmul__(self, tensor:'Tensor') -> 'Tensor':
        tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor, device=self.device)
        return F.matmul(self, tensor)
    
    
    def __rmatmul__(self, tensor:'Tensor') -> 'Tensor':
        tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor, device=self.device)
        return F.matmul(tensor, self)
    
    
    def __pow__(self, power) -> 'Tensor':
        return F.pow(self, power)

    def __rpow__(self, power) -> 'Tensor':
        return F.rpow(self, power)

    def __neg__(self) -> 'Tensor': # -self
        return self * -1.0

    def __radd__(self, other) -> 'Tensor': # other + self
        return self + other

    def __sub__(self, other) -> 'Tensor': # self - other
        return self + (-other)

    def __rsub__(self, other) -> 'Tensor': # other - self
        return other + (-self)

    def __rmul__(self, other) -> 'Tensor': # other * self
        return self * other

    def __truediv__(self, other) -> 'Tensor': # self / other
        return self * other**-1

    def __rtruediv__(self, other) -> 'Tensor': # other / self
        return other * self**-1
    
    def __getitem__(self, key) -> 'Tensor':
        return F.slice(self, key)
    
    def __iter__(self):
        self._current_idx = 0
        return self
    
    def __next__(self) -> 'Tensor':
        if self._current_idx >= len(self):
            raise StopIteration
        else:
            val = self[self._current_idx]
            self._current_idx += 1
            return val
    
    def __len__(self) -> int:
        return len(self.data)
    
    # *************************
    # ******* Other ops *******
    # *************************
    
    def exp(self) -> 'Tensor':
        return F.exp(self)
    
    def log(self) -> 'Tensor':
        return F.log(self)
    
    def sqrt(self) -> 'Tensor':
        return F.sqrt(self)
    
    def sum(self, dim:int=None, keepdims=False) -> 'Tensor':
        return F.sum(self, dim, keepdims)
    
    def mean(self, dim:int=None, keepdims=False) -> 'Tensor':        
        return F.mean(self, dim, keepdims)

    def max(self, dim:int=None, keepdims=False) -> 'Tensor': 
        return F.max(self, dim, keepdims)
    
    def min(self, dim:int=None, keepdims=False) -> 'Tensor':
        return F.min(self, dim, keepdims)
    
    def squeeze(self, dim:'int | tuple[int]'=None) -> 'Tensor':
        return F.squeeze(self, dim)
    
    def unsqueeze(self, dim) -> 'Tensor':
        return F.unsqueeze(self, dim)
    
    def reshape(self, shape:tuple) -> 'Tensor':
        return F.reshape(self, shape)
    
    def transpose(self, dim0:int, dim1:int) -> 'Tensor':
        """ Transpose tensor, dim0 and dim1 are swapped (0, 1 for 2D tensor)"""    
        return F.transpose(self, dim0, dim1)
     
    def flatten(self, start_dim=0, end_dim=-1) -> 'Tensor':
        return F.flatten(self, start_dim, end_dim)
    
    def unfold(self, dimension:int, size:int, step:int) -> 'Tensor':
        return F.unfold_dim(self, dimension, size, step)
    
    # *********************
    # **** Pretty data ****
    # *********************
    
    def __str__(self) -> str:
        return self.__repr__()
        
    def __repr__(self) -> str:
        pretty_data_str = tools.pretty_numpy(self.data, separator=", ")
        beggining = "Tensor("
        data_str = ""
        for i, line in enumerate(pretty_data_str.splitlines(keepends=True)):
            if i != 0:
                line = " "*len(beggining) + line
            data_str += line
        string = f"{beggining}{data_str}, shape={self.shape}"
        
        if self.requires_grad:
            if self.grad_fn is not None:
                string += f", grad_fn=<{self.grad_fn.name()}>"
            else:
                string += f", requires_grad={self.requires_grad}"
            
        if self._name is not None:
            string += f", name={self._name}"
            
            
        string += f", dtype={self.dtype})"
            
        return string