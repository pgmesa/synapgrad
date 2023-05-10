import random
import numpy as np
import importlib
from typing import Union

from ..device import Device
from ..utils import tools

        
default_type__ = np.float32


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
            >>> Tensor(np.array([1, 2, 3]), dtype=np.float32, requires_grad=True, name="weights", device=Device.GPU)
            >>> Tensor([0.0], dtype=np.float64, requires_grad=True, name="bias", device=Device.GPU)
            
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=default_type__)
        if dtype is not None and data.dtype != dtype: data = data.astype(dtype)
        assert isinstance(data, np.ndarray), "data must be a list or numpy array"
        
        # Import inside tensor in order to avoid circular import issue
        self.F = importlib.import_module('.autograd_functions', 'synapgrad.tensor')
        self.autograd = importlib.import_module('.autograd', 'synapgrad')
        
        self.data = data
        self.device = device if device is not None else Device.CPU
        # Internal variables used for autograd graph construction
        self._grad = None
        self._grad_fn = None
        req_grad = requires_grad and self.autograd.gradient__
        if req_grad and not self.is_floating_point(data):
            raise RuntimeError("Only floating point Tensors can require gradients")
        self._requires_grad = req_grad
        self._is_leaf = True
        self._retain_grad = False
        self._children = children
        self._operation = operation # Operation that produced this node, for graphviz / debugging
        self._name = name
        
    # **************************
    # ******* Properties *******
    # ************************** 
    
    @property
    def name(self) -> str:
        return "" if self._name is None else str(self._name)    
        
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
        
        if value and not self.is_floating_point(self.data):
            raise RuntimeError("Only floating point Tensors can require gradients")
        
        self._requires_grad = value
    
    @property
    def grad(self) -> 'Tensor':
        if not self._is_leaf and not self._retain_grad:
            print("\n[!] WARNING: The .grad attribute of a Tensor that is not a " + 
                  "leaf Tensor is being accessed. Its .grad attribute won't be populated " + 
                  "during autograd.backward(). If you indeed want the .grad field to be populated " + 
                  "for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor")
        return Tensor(self._grad) if self._grad is not None else None

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
        if not self.requires_grad:
            raise RuntimeError("Cannot set grad_fn for a Tensor that doesn't require grad")
        self._grad_fn = grad_fn
    
    # *********************************
    # ******* Utility functions *******
    # *********************************
    
    def numel(self) -> int:
        return self.data.size
    
    def retain_grad(self):
        """ Grad is not stored in not leaf tensors by default to avoid extra memory consumption. Call
        this function to enable grad storing"""
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
        return self.F.clone(self)
    
    def contiguous(self) -> 'Tensor':
        """
        Returns a contiguous tensor.

        Returns
        -------
        Tensor
            The contiguous tensor.
        """
        out = Tensor(np.ascontiguousarray(self.data), (self,), '<Contiguous>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += out._grad
            
        out._backward = _backward
        
        return out
    
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
                grad = Tensor(1.0, dtype=self.dtype)
        
        assert self.is_floating_point(grad), "expected float dtype for grad, got %s" % grad.dtype
          
        if not isinstance(grad, Tensor):
            raise ValueError("Gradient parameter must be a Tensor")
        
        self.autograd.backward(self.grad_fn, grad)
        
        
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
        return  self.F.add(self, summand)
        
        
    def __mul__(self, factor:'Tensor') -> 'Tensor':
        factor = factor if isinstance(factor, Tensor) else Tensor(factor, device=self.device)
        from . import functional as F
        return self.F.mul(self, factor)
    
    
    def __matmul__(self, tensor:'Tensor') -> 'Tensor':
        tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor, device=self.device)
        return self.F.matmul(self, tensor)
    
    
    def __rmatmul__(self, tensor:'Tensor') -> 'Tensor':
        tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor, device=self.device)
        return self.F.matmul(tensor, self)
    
    
    def __pow__(self, power) -> 'Tensor':
        return self.F.pow(self, power)

    def __rpow__(self, power) -> 'Tensor':
        return self.F.rpow(self, power)

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
        return self.F.slice(self, key)
    
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
        return np.e**self
    
    
    def log(self) -> 'Tensor':
        out = Tensor(np.log(self.data), (self,), '<Log>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += (out._grad / (self.data + 1e-10)) 
                
        out._backward = _backward
        
        return out
    
    
    def sqrt(self) -> 'Tensor':
        out = Tensor(np.sqrt(self.data), (self,), '<Sqrt>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += out._grad / (2 * out.data)
                
        out._backward = _backward
        
        return out
    
    
    def sum(self, dim:int=None, keepdims=False) -> 'Tensor':
        out = Tensor(self.data.sum(axis=dim, keepdims=keepdims), (self,), '<Sum>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                out_grad = out._grad
                if not keepdims and dim is not None:
                    s = list(self.shape)
                    for d in dim: s[d] = 1
                    out_grad = out._grad.reshape(s) 
                self._grad += np.ones(self.shape) * out_grad
            
        out._backward = _backward
        
        return out
    
    
    def mean(self, dim:int=None, keepdims=False) -> 'Tensor':
        out = Tensor(self.data.mean(axis=dim, keepdims=keepdims), (self,), '<Mean>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                if dim is None:
                    size = self.data.size
                elif isinstance(dim, int):
                    size = self.data.shape[dim]
                elif isinstance(dim, tuple):
                    size = 1
                    for d in dim: size *= self.data.shape[d]
                out_grad = out._grad
                if not keepdims and dim is not None:
                    s = list(self.shape)
                    for d in dim: s[d] = 1
                    out_grad = out._grad.reshape(s) 
                self._grad += (np.ones(self.shape) / size) * out_grad
            
        out._backward = _backward
        
        return out

    
    def max(self, dim:int=None, keepdims=False, return_indices=None, return_selected=False) -> tuple['Tensor',...]:
        max_values = self.data.max(axis=dim, keepdims=keepdims)
        max_indices = self.data.argmax(axis=dim)
        selected = tools.get_selected_from_indices(self.data, max_indices, dim)
        
        out = Tensor(max_values, (self,), '<Max>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                grad = out._grad
                if not keepdims and dim is not None:
                    grad = np.expand_dims(grad, axis=dim)
                self._grad += selected * grad
            
        out._backward = _backward
        
        out_tuple = (out,)
        
        if return_indices is not False and dim is not None:
            out_tuple += (max_indices,)
        
        if return_selected:
            out_tuple += (selected,)
        
        return out_tuple if len(out_tuple) > 1 else out_tuple[0]
    
    
    def min(self, dim:int=None, keepdims=False, return_indices=None, return_selected=False) -> tuple['Tensor',...]:
        min_values = self.data.min(axis=dim, keepdims=keepdims)
        min_indices = self.data.argmin(axis=dim)
        selected = tools.get_selected_from_indices(self.data, min_indices, dim)
        
        out = Tensor(min_values, (self,), '<Min>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                grad = out._grad
                if not keepdims and dim is not None:
                    grad = np.expand_dims(grad, axis=dim)
                self._grad += selected * grad
            
        out._backward = _backward
        
        out_tuple = (out,)
        
        if return_indices is not False and dim is not None:
            out_tuple += (min_indices,)
        
        if return_selected:
            out_tuple += (selected,)
        
        return out_tuple if len(out_tuple) > 1 else out_tuple[0]

    
    def reshape(self, shape:tuple) -> 'Tensor':
        out = Tensor(self.data.reshape(shape), (self,), '<Reshape>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += out._grad.reshape(self.shape)
            
        out._backward = _backward
        
        return out
    
    
    def view(self, *shape:tuple) -> 'Tensor':
        out = Tensor(self.data.reshape(shape), (self,), '<View>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += out._grad.reshape(self.shape)
            
        out._backward = _backward
        
        return out
    
    
    def squeeze(self, dim:int=None) -> 'Tensor':
        data = self.data
        can_apply = len(self.shape) > 0 and (dim is None or self.shape[dim] == 1)
        if can_apply:
            data = np.squeeze(self.data, dim)
            if dim is None:
                dim = []
                for i, d in enumerate(self.data.shape):
                    if d == 1: dim.append(i) 
        out = Tensor(data, (self,), '<Squeeze>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                if can_apply:
                    self._grad += np.expand_dims(out._grad, dim)
                else:
                    self._grad += out._grad
            
        out._backward = _backward
        
        return out
    
    
    def unsqueeze(self, dim) -> 'Tensor':
        data = np.expand_dims(self.data, dim)
        out = Tensor(data, (self,), '<Unsqueeze>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += np.squeeze(out._grad, dim)
            
        out._backward = _backward
        
        return out
    
    
    def flatten(self, start_dim=0, end_dim=-1) -> 'Tensor':
        shape = self.shape
        start = start_dim if start_dim != -1 else len(shape)
        end = end_dim if end_dim != -1 else len(shape)
        if start > end:
            raise RuntimeError("flatten() has invalid args: start_dim cannot come after end_dim")
        if start < end:
            shape = self.shape[:start] + (-1,) + self.shape[end+1:]
        out = Tensor(self.data.reshape(shape), (self,), '<Flatten>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += out._grad.reshape(self.shape)
                
        out._backward = _backward
        
        return out
    
    
    def unfold(self, dimension:int, size:int, step:int) -> 'Tensor':
        """
        Unfold a tensor along a specified dimension.

        Parameters
        ----------
        tensor : Tensor
            The tensor to unfold.
        dimension : int
            The dimension to unfold.
        size : int
            The size of the unfolding window.
        step : int
            The step between each unfolding window.

        Returns
        -------
        Tensor
            The unfolding result.

        Raises
        ------
        ValueError
            If the specified dimension is invalid, if the size or step are not positive integers, or if the size is
            larger than the size of the specified dimension.

        Examples
        --------
        >>> import numpy as np
        >>> a = np.arange(10)
        >>> unfold(a, 0, 3, 1)
        array([[[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]],
            
               [[9, 0, 1],
                [2, 3, 4],
                [5, 6, 7]]])
        """
        tensor = self.data
        # check that the specified dimension is valid
        if dimension >= tensor.ndim or dimension < -tensor.ndim:
            raise ValueError(f"Dimension out of range for tensor with {tensor.ndim} dimensions: {dimension}")
        if dimension < 0:
            dimension += tensor.ndim
        # check that the size and step are positive integers
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"Invalid size: {size}")
        if not isinstance(step, int) or step <= 0:
            raise ValueError(f"Invalid step: {step}")
        # get the size of the specified dimension
        dim_size = tensor.shape[dimension]
        # check that the size is smaller than or equal to the size of the dimension
        if size > dim_size:
            raise ValueError(f"Size ({size}) must be smaller than or equal to the size of the specified dimension ({dim_size})")
        # calculate the size of the output dimension
        out_size = int((dim_size - size) / step) + 1
        # create an output array with the appropriate shape
        out_shape = list(tensor.shape)
        out_shape[dimension] = out_size
        out_shape.append(size)
        out_array = np.zeros(out_shape, dtype=tensor.dtype)
        # fill the output array with the unfolded slices
        for i in range(out_size):
            start = i * step
            end = start + size
            window = np.take(tensor, np.arange(start, end), axis=dimension)
            window = np.moveaxis(window, dimension, -1)
            out_array = np.delete(out_array, i, axis=dimension)
            out_array = np.insert(out_array, i, window, axis=dimension)
        
        out = Tensor(out_array, (self,), f'<UnfoldDim{dimension}>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                folded = np.zeros_like(tensor)
                for i in range(out._grad.shape[dimension]):
                    start = i * step
                    end = start + size
                    s1 = [slice(None)] * (dimension + 1); s1[dimension] = slice(start, end)
                    s2 = [slice(None)] * (dimension + 1); s2[dimension] = i
                    s1 = tuple(s1); s2 = tuple(s2)
                    folded[s1] += np.moveaxis(out._grad[s2], -1, dimension).reshape(folded[s1].shape)
                
                self._grad += folded 
            
        out._backward = _backward

        return out    
    
    def transpose(self, dim0:int, dim1:int) -> 'Tensor':
        """ Transpose tensor, dim0 and dim1 are swapped (0, 1 for 2D tensor)"""
        out = Tensor(self.data.swapaxes(dim0, dim1), (self,), '<Transpose>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += out._grad.swapaxes(dim0, dim1)
                
        out._backward = _backward
        
        return out
    
    # *********************
    # **** Pretty data ****
    # *********************
    
    def __str__(self) -> str:
        return self.__repr__()
        
    def __repr__(self) -> str:
        pretty_data_str = tools.pretty_numpy(self.data)
        beggining = "Tensor("
        data_str = ""
        for i, line in enumerate(pretty_data_str.splitlines(keepends=True)):
            if i != 0:
                line = " "*len(beggining) + line
            data_str += line
        string = f"{beggining}{data_str}, shape={self.shape}"
        
        if self.requires_grad:
            if self.grad_fn is not None:
                string += f", grad_fn={self.grad_fn}"
            else:
                string += f", requires_grad={self.requires_grad}"
            
        if self._name is not None:
            string += f", name={self._name}"
            
            
        string += f", dtype={self.dtype})"
            
        return string