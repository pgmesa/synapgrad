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
    
    def __init__(self, data, _children=(), _operation=None, requires_grad=False, dtype=None) -> None:
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=default_type__)
        if dtype is not None and data.dtype != dtype: data = data.astype(dtype)
        assert isinstance(data, np.ndarray), "data must be a list or numpy array"
        
        self.data = data
        # Internal variables used for autograd graph construction
        self._grad = None
        self._grad_fn = None
        self._requires_grad = requires_grad and gradient__
        self._is_leaf = True
        self._retain_grad = False
        self._children = _children
        self._operation = _operation # Operation that produced this node, for graphviz / debugging         
    
    
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
                print(tdim, gdim)
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
                tensor += grad.sum(axis=not_compatible_dims, keepdims=True)
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
                self._grad += np.dot(tensor.data, out._grad.T).T
            
            if tensor.requires_grad:
                tensor._grad += np.dot(self.data.T, out._grad)
                    
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
    
    
    @staticmethod
    def add(t1:'Tensor', t2:'Tensor') -> 'Tensor':
        return t1 + t2
    
    
    @staticmethod
    def mul(t1:'Tensor', t2:'Tensor') -> 'Tensor':
        return t1 * t2
    
    
    @staticmethod
    def matmul(t1:'Tensor', t2:'Tensor') -> 'Tensor':
        return t1 @ t2
    
    
    def view(self, shape:tuple) -> 'Tensor':
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
    
    
    def sum(self, dim:int=None, keepdims=False) -> 'Tensor':
        out = Tensor(self.data.sum(axis=dim, keepdims=keepdims), (self,), '<Sum>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += out._grad
            
        out._backward = _backward
        
        return out
    
    
    def mean(self, dim:int=None) -> 'Tensor':
        out = Tensor(self.data.mean(axis=dim), (self,), '<Mean>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += (np.ones(self.shape) / self.data.size) * out._grad
            
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
    
    
    def backward(self, grad:np.ndarray=None):
        if not self.requires_grad:
            raise RuntimeError("Trying to call backward on Tensor with requires_grad=False")

        if grad is None:
            if self.data.size > 1:
                raise RuntimeError("grad must be specified for non-scalar tensors")
            else:
                grad = np.array(1.0, dtype=self.data.dtype)
                
        if not isinstance(grad, np.ndarray):
            raise ValueError("Gradient parameter must be a numpy array")
                
        self._is_leaf = True
        
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
    def pretty_numpy(array:np.ndarray, decimals=5) -> str:
        def truncate(f, n):
            return np.floor(f * 10 ** n) / 10 ** n
        rounded_data = array.copy().round(decimals=decimals)
        #rounded_data += 0. # Remove -0.0 values (just 0.0)
        str_to_rm = "array("
        data_str = repr(rounded_data).replace(str_to_rm, "")
        crop_index = (len(data_str)) - data_str[::-1].find("]")
        cropped = data_str[:crop_index]
        
        braces_padd = (len(array.shape)-1)
        no_padding = cropped.replace("\n" + " "*(braces_padd+len(str_to_rm)), "\n" + " "*braces_padd)
        return no_padding
        
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
            string += f", op={self._operation})"
            
        string += ")"
            
        return string