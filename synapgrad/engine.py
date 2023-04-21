import random
import numpy as np
from typing import Iterable, Union


gradient__ = True
retain_grads__ = False

default_type__ = np.float32

class no_grad:
    
    def __init__(self) -> None:
        self.prev = gradient__
    
    def __enter__(self):
        global gradient__
        gradient__ = False
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        global gradient__
        gradient__ = self.prev
        

class retain_grads:
    def __init__(self) -> None:
        self.prev = retain_grads__
    
    def __enter__(self):
        global retain_grads__
        retain_grads__ = True
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        global retain_grads__
        retain_grads__ = self.prev
        

def manual_seed(seed:int):
    np.random.seed(seed)
    random.seed(seed)


def tensor(data, requires_grad=False, dtype=None) -> 'Tensor':
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


class Tensor:
    
    def __init__(self, data, _children=(), _operation=None, requires_grad=False, dtype=None, name=None) -> None:
        """
        Creates a Tensor object from the given data, which is always transformed internally into a numpy array.

        Args:
            data (number or iterable): data of the tensor, must be convertible into a numpy.array().
            _children (tuple, optional): tensors which produced this tensor as a result of an operation. Defaults to ().
            _operation (str, optional): string that represents the operation that created this tensor. Defaults to None.
            requires_grad (bool, optional): whether this tensor requieres gradients or not. Defaults to False.
            dtype (type, optional): numpy type of this tensor data. Defaults to None.
            
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=default_type__)
        if dtype is not None and data.dtype != dtype: data = data.astype(dtype)
        assert isinstance(data, np.ndarray), "data must be a list or numpy array"
        
        self.data = data
        # Internal variables used for autograd graph construction
        self._grad = None
        self._grad_fn = None
        req_grad = requires_grad and gradient__
        if req_grad and not self.is_floating_point(data):
            raise RuntimeError("Only floating point Tensors can require gradients")
        self._requires_grad = req_grad
        self._is_leaf = True
        self._retain_grad = False
        self._children = _children
        self._operation = _operation # Operation that produced this node, for graphviz / debugging
        self._name = name        
    
    
    @staticmethod
    def __add_grad(tensor:'Tensor', grad:np.ndarray):
        """Function that ensures that the gradient that wants to be added is compatible
        with the tensor gradient. It is a useful function for operations like __add__ and __mul__
        when opertions like v1:(3,) (+ or *) v2:(2,3) pop up. In this case the incomming gradient
        would be shape (2,3) and the gradient of v1 is (3,), so a sum() operation over axis 0 is required.
        This function calculates the necessary axes in which to perform the sum() operation.

        Args:
            tensor (Tensor): Tensor whose gradient is going to be updated
            grad (np.ndarray): Gradient array to add
        """
        def get_incompatible_dims(tensor_shape, grad_shape) -> tuple:
            not_compatible_dims = []
            for i, (tdim, gdim) in enumerate(zip(tensor_shape[::-1], grad_shape[::-1])):
                if gdim != 1 and tdim != gdim:
                    not_compatible_dims.append(len(grad_shape)-1-i)
            not_compatible_dims = tuple(not_compatible_dims)
            return not_compatible_dims
            
        
        if len(tensor.data.squeeze().shape) == 0:
            tensor._grad += grad.sum()
            return
        
        tensor_shape = tensor.data.shape; grad_shape = grad.data.shape
        diff_axis = len(grad_shape)-len(tensor_shape)
        sum_axis = None if diff_axis <= 0 else tuple(np.arange(abs(diff_axis), dtype=np.int8).tolist())
        
        if tensor.grad.matches_shape(grad) or sum_axis is None: 
            if sum_axis is None and not tensor.grad.matches_shape(grad):
                not_compatible_dims = get_incompatible_dims(tensor_shape, grad_shape)
                tensor._grad += grad.sum(axis=not_compatible_dims, keepdims=True)
            else:
                tensor._grad += grad
        else:
            tensor._grad += grad.sum(axis=sum_axis)
    
    
    def __add__(self, summand:'Tensor') -> 'Tensor':
        summand = summand if isinstance(summand, Tensor) else Tensor(summand)
        r_grad = self.requires_grad or summand.requires_grad
        
        out = Tensor(self.data + summand.data, (self, summand), '<Add>', requires_grad=r_grad)
        
        def _backward():
            if self.requires_grad:
                self.__add_grad(self, out._grad)
                
            if summand.requires_grad:
                self.__add_grad(summand, out._grad)
    
        out._backward = _backward 
        
        return out
        
        
    def __mul__(self, factor:'Tensor') -> 'Tensor':
        factor = factor if isinstance(factor, Tensor) else Tensor(factor)
        r_grad = self.requires_grad or factor.requires_grad
        out = Tensor(self.data * factor.data, (self, factor), '<Mul>', requires_grad=r_grad)
        
        def _backward():
            if self.requires_grad:
                self.__add_grad(self, factor.data * out._grad)
            
            if factor.requires_grad:
                self.__add_grad(factor, self.data * out._grad)
            
        out._backward = _backward 
        
        return out
    
    
    def __matmul__(self, tensor:'Tensor') -> 'Tensor':
        tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
        r_grad = self.requires_grad or tensor.requires_grad
        assert len(self.data.shape) > 1 and len(tensor.data.shape) > 1, "Both inputs should have dimension > 1"
        out = Tensor(self.data @ tensor.data, (self, tensor), '<Matmul>', requires_grad=r_grad)
        
        def _backward():
            if self.requires_grad:
                grad = out._grad @ np.moveaxis(tensor.data, -2, -1)
                self.__add_grad(self, grad)
            
            if tensor.requires_grad:
                grad = np.moveaxis(out._grad, -2,-1) @ self.data
                grad = np.moveaxis(grad, -2,-1)
                self.__add_grad(tensor, grad)
                    
        out._backward = _backward
        
        return out
    
    
    def __rmatmul__(self, tensor:'Tensor') -> 'Tensor':
        tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
        r_grad = self.requires_grad or tensor.requires_grad
        assert len(self.data.shape) > 1 and len(tensor.data.shape) > 1, "Both inputs should have dimension > 1"
        out = Tensor(tensor.data @ self.data, (self, tensor), '<Matmul>', requires_grad=r_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += np.dot(tensor.data.T, out._grad)
            
            if tensor.requires_grad:
                tensor._grad += np.dot(self.data, out._grad.T).T
                    
        out._backward = _backward
    
    
    def __pow__(self, power) -> 'Tensor':
        assert isinstance(power, (int, float)), f"Power of type '{type(power)}' not supported"
        out = Tensor(self.data**power, (self,), f'<Pow({power})>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += (power * self.data**(power-1)) * out._grad
                    
        out._backward = _backward
        
        return out
    
    
    def __rpow__(self, power) -> 'Tensor':
        assert isinstance(power, (int, float)), f"Power of type '{type(power)}' not supported"
        out = Tensor(np.power(power, self.data), (self,), f'<rPow({power})>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += (power**self.data * np.log(power)) * out._grad
                    
        out._backward = _backward
        
        return out
    
    
    def __getitem__(self, key) -> 'Tensor':
        new_data = self.data[key]
        out = Tensor(new_data, (self,), _operation='<Slice>', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self._grad[key] = out._grad
        
        out._backward = _backward

        return out
    
    
    @staticmethod
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
    
    
    @staticmethod
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
    
    
    def add(self, t2:'Tensor') -> 'Tensor':
        return self + t2
    
    
    def mul(self, t2:'Tensor') -> 'Tensor':
        return self * t2
    
    
    def matmul(self, t2:'Tensor') -> 'Tensor':
        return self @ t2
    
    
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
    
    
    @staticmethod
    def __get_selected_from_indices(values:np.ndarray, indices:np.ndarray, dim) -> np.ndarray:
        """
        Returns a boolean array of the same shape as the input array, where each element is True if the corresponding
        element in the input array is the selected value in the corresponding dimension.

        Parameters
        ----------
        values : np.ndarray
            The values in the input array.
        indices : np.ndarray
            The indices of the selected values in the input array.
        dim : int
            The dimension of the input array.

        Returns
        -------
        np.ndarray
            The boolean array.
        """
        def get_indices(array, dim):
            indices = []
            if dim == -1: dim = array.ndim - 1
            for i in range(dim):
                s = [None,] * dim
                s[i] = slice(None)
                s = tuple(s)
                #print(s)
                indices.append(np.arange(array.shape[i])[s])
            return tuple(indices)
    
        selected = np.zeros_like(values)
        
        if dim is not None:
            slices = get_indices(values, dim)
            slices = list(slices)
            slices.append(indices)
            slices = tuple(slices)
        else:
            slices = np.unravel_index(indices, values.shape)

        selected[slices] = 1
        
        return selected
    
    
    def sum(self, dim:int=None, keepdims=False) -> 'Tensor':
        out = Tensor(self.data.sum(axis=dim, keepdims=keepdims), (self,), '<Sum>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += out._grad
            
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
                self._grad += (np.ones(self.shape) / size) * out._grad
            
        out._backward = _backward
        
        return out

    
    def max(self, dim:int=None, keepdims=False, return_indices=None, return_selected=False) -> tuple['Tensor',...]:
        max_values = self.data.max(axis=dim, keepdims=keepdims)
        max_indices = self.data.argmax(axis=dim)
        selected = self.__get_selected_from_indices(self.data, max_indices, dim)
        
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
        selected = self.__get_selected_from_indices(self.data, min_indices, dim)
        
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
    
    
    def transpose(self, dim0:int, dim1:int) -> 'Tensor':
        """ Transpose tensor, dim0 and dim1 are swapped (0, 1 for 2D tensor)"""
        out = Tensor(self.data.swapaxes(dim0, dim1), (self,), '<Transpose>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += out._grad.swapaxes(dim0, dim1)
                
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
    
    
    @staticmethod
    def is_floating_point(array) -> bool:
        return array.dtype == np.float16 or array.dtype == np.float32 or array.dtype == np.float64
    
    
    def backward(self, grad:np.ndarray=None):
        if not self.requires_grad:
            raise RuntimeError("Trying to call backward on Tensor with requires_grad=False")

        if grad is None:
            if self.data.size > 1:
                raise RuntimeError("grad must be specified for non-scalar tensors")
            else:
                grad = np.array(1.0, dtype=self.data.dtype)
        
        assert self.is_floating_point(grad), "expected float dtype for grad, got %s" % grad.dtype
          
        if not isinstance(grad, np.ndarray):
            raise ValueError("Gradient parameter must be a numpy array")
        
        # Topological order all of the children in the graph 
        # (init gradients for those who are going to need it)
        ordered_nodes = []
        visited_nodes = set()
        def visit_node(node):
            if node not in visited_nodes:
                visited_nodes.add(node)
                for child in node._children:
                    if child.requires_grad and child._grad is None:
                        child.zero_()
                    visit_node(child)
                ordered_nodes.append(node)
        visit_node(self)

        # Go one tensor at a time and apply the chain rule to get its gradient
        self._grad = grad
        for i, node in enumerate(reversed(ordered_nodes)):
            if node.grad_fn is not None:
                node.grad_fn()
            if node is not self and not node.is_leaf and not node._retain_grad and not retain_grads__:
                del node._grad
                node._grad = None
    
    def zero_(self):
        self._grad = np.zeros_like(self.data)
    
    def retain_grad(self):
        """ Grad is not stored in not leaf tensors by default to avoid extra memory consumption. Call
        this function to enable grad storing"""
        self._retain_grad = True
        
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
        return Tensor(self.data)
    
    def cpu(self) -> 'Tensor':
        """ Returns itself, just added for compatibility with .cpu() pytorch call """
        return self
        
    def matches_shape(self, tensor:Union['Tensor', np.ndarray]) -> bool:
        if len(self.shape) != len(tensor.shape):
            return False
        
        for n1, n2 in zip(self.shape, tensor.shape):
            if n1 != n2: return False
        
        return True
        
        
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
    
    @property
    def grad_fn(self) -> 'Tensor':
        return self._grad_fn
    
    @grad_fn.setter
    def _backward(self, grad_fn):
        if gradient__ and self.requires_grad:
            self._grad_fn = grad_fn
    
    
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
    
    def __neg__(self) -> 'Tensor': # -self
        return self * -1

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
    
    @staticmethod
    def pretty_numpy(array:np.ndarray, precision=4, separator=',') -> str: 
        data_str = np.array2string(array, precision=precision, separator=separator)
        return data_str
        
    def __repr__(self) -> str:
        pretty_data_str = self.pretty_numpy(self.data)
        beggining = "Tensor("
        data_str = ""
        for i, line in enumerate(pretty_data_str.splitlines(keepends=True)):
            if i != 0:
                line = " "*len(beggining) + line
            data_str += line
        string = f"{beggining}{data_str}, shape={self.shape}"
        
        if self.requires_grad:
            string += f", req_grad={self.requires_grad}"
            
        if self._operation is not None:
            string += f", op={self._operation}"
            
        if self._name is not None:
            string += f", name={self._operation}"
            
            
        string += f", dtype={self.dtype})"
            
        return string