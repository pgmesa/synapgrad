
import random
import numpy as np
from typing import Iterable


gradient__ = True


class no_grad:
    
    def __init__(self) -> None:
        self.prev = gradient__
    
    def __enter__(self):
        global gradient__
        gradient__ = False
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        global gradient__
        gradient__ = self.prev
        

def manual_seed(seed:int):
    np.random.seed(seed)
    random.seed(seed)


def tensor(data, requires_grad=False) -> 'Tensor':
    return Tensor(data, requires_grad=requires_grad)


class Tensor:
    
    def __init__(self, data, _children=(), _operation='None', requires_grad=False) -> None:
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)
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
    
    
    def __add__(self, summand:'Tensor') -> 'Tensor':
        summand = summand if isinstance(summand, Tensor) else Tensor(summand)
        r_grad = self.requires_grad or summand.requires_grad
        out = Tensor(self.data + summand.data, (self, summand), '<Add>', requires_grad=r_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += out._grad.sum() if len(self.data.shape) == 0 else out._grad
                
            if summand.requires_grad:
                summand._grad += out._grad.sum() if len(summand.data.shape) == 0 else out._grad
    
        out._backward = _backward 
        
        return out
        
        
    def __mul__(self, factor:'Tensor') -> 'Tensor':
        factor = factor if isinstance(factor, Tensor) else Tensor(factor)
        r_grad = self.requires_grad or factor.requires_grad
        out = Tensor(self.data * factor.data, (self, factor), '<Mul>', requires_grad=r_grad)
        
        def _backward():
            if self.requires_grad:
                self_grad = factor.data * out._grad
                self._grad += self_grad.sum() if len(self.data.shape) == 0 else self_grad
            
            if factor.requires_grad:
                factor_grad = self.data * out._grad
                factor._grad += factor_grad.sum() if len(factor.data.shape) == 0 else factor_grad
            
        out._backward = _backward 
        
        return out
    
    
    def __matmul__(self, tensor:'Tensor') -> 'Tensor':
        tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
        r_grad = self.requires_grad or tensor.requires_grad
        out = Tensor(np.matmul(self.data, tensor.data), (self, tensor), '<Matmul>', requires_grad=r_grad)
        
        def _backward():
            if self.requires_grad:
                self_grad = tensor.data * out._grad
                self._grad += self_grad.sum() if len(self.data.shape) == 0 else self_grad
            
            if tensor.requires_grad:
                tensor_grad = self.data * out._grad
                tensor._grad += tensor_grad.sum() if len(tensor.data.shape) == 0 else tensor_grad
                    
        out._backward = _backward
        
        return out
    
    
    def __pow__(self, power) -> 'Tensor':
        assert isinstance(power, (int, float)), f"Power of type '{type(power)}' not supported"
        out = Tensor(self.data**power, (self,), f'<Pow({power})>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self_grad = (power * self.data**(power-1)) * out._grad
                self._grad += self_grad.sum() if len(self.data.shape) == 0 else self_grad
                    
        out._backward = _backward
        
        return out
    
    
    def __rpow__(self, power) -> 'Tensor':
        assert isinstance(power, (int, float)), f"Power of type '{type(power)}' not supported"
        out = Tensor(np.power(power, self.data), (self,), f'<rPow({power})>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self_grad = (power**self.data * np.log(power)) * out._grad
                self._grad += self_grad.sum() if len(self.data.shape) == 0 else self_grad
                    
        out._backward = _backward
        
        return out
    
    
    def __getitem__(self, key) -> 'Tensor':
        new_data = self.data[key]
        out = Tensor(new_data, (self,), _operation='Slice', requires_grad=self.requires_grad)

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
    
    
    def view(self, shape:tuple) -> 'Tensor':
        out = Tensor(self.data.reshape(shape), (self,), '<View>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += out._grad.reshape(self.shape)
            
        out._backward = _backward
        
        return out
    
    
    def squeeze(self, dim=None) -> 'Tensor':
        data = np.squeeze(self.data, dim)
        if dim is None:
            dim = []
            for i, d in enumerate(self.data.shape):
                if d == 1: dim.append(i)
                
        out = Tensor(data, (self,), '<Squeeze>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += np.expand_dims(out._grad, dim)
            
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
    
    
    def sum(self, dim:int=None) -> 'Tensor':
        out = Tensor(self.data.sum(axis=dim), (self,), '<Sum>', requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self._grad += np.ones(self.shape) * out._grad
            
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
    
    
    def backward(self, grad:np.ndarray=None):
        if not self.requires_grad:
            raise RuntimeError("Trying to call backward on Tensor with requires_grad=False")

        if grad is None:
            if self.data.size > 1:
                raise RuntimeError("grad must be specified for non-scalar tensors")
            else:
                grad = np.array(1.0, dtype=np.float64)
                
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
            if node is not self and not node.is_leaf and not node._retain_grad:
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
        
    def matches_shape(self, tensor:'Tensor') -> bool:
        if len(self.shape) != len(tensor.shape):
            return False
        
        for n1, n2 in zip(self.shape, tensor.shape):
            if n1 != n2: return False
        
        return True
                
    @property
    def shape(self) -> tuple:
        return self.data.shape
    
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
        
    def __repr__(self) -> str:
        return f"Tensor({self.data}, shape={self.shape}, req_grad={self.requires_grad}, op={self._operation})"


# Check with pytorch that gradients are correct when applying different tensor operations
if __name__ == "__main__":
    l1 = [[-4.0, 0, 5.0], [6.3, 3.2, 1.3]]
    l2 = [[2.0, 2,  3.0], [2.4, 1.7, 0.5]]
    
    a = Tensor(l1, requires_grad=True).unsqueeze(0)
    a.retain_grad()
    b = Tensor(l2, requires_grad=True).unsqueeze(0)
    b.retain_grad()
    c = Tensor(4.0, requires_grad=True)
    c.retain_grad()
    out = Tensor.concat((a*c, b), dim=1).transpose(0, 1)[0, :]
    out = out.view(3).unsqueeze(0).unsqueeze(0).squeeze()
    s = out.sum()
    s.backward()
    
    print("Tensor 1: ", a)
    print("Tensor 2: ", b)
    print("Tensor 3: ", c)
    print("Tensor resultado: ", out)
    print("Gradiente de tensor 1: ", a.grad)
    print("Gradiente de tensor 2: ", b.grad)
    print("Gradiente de tensor 3: ", c.grad)
    
    import torch
    # Creamos el tensor con valores diferentes
    a = torch.tensor(l1, requires_grad=True).unsqueeze(0)
    a.retain_grad()
    b = torch.tensor(l2, requires_grad=True).unsqueeze(0)
    b.retain_grad()
    c = torch.tensor(4.0, requires_grad=True)
    c.retain_grad()

    # Realizamos la multiplicación entre los dos tensores
    out = torch.concat((a*c, b), dim=1).transpose(0, 1)[0, :]
    out = out.view(3).unsqueeze(0).unsqueeze(0).squeeze()

    # Calculamos el gradiente del tensor resultado
    s = out.sum()
    s.backward()

    # Mostramos los resultados
    print("\nTensor 1: ", a)
    print("Tensor 2: ", b)
    print("Tensor 3:", c)
    print("Tensor resultado: ", out)
    print("Gradiente de tensor 1: ", a.grad)
    print("Gradiente de tensor 2: ", b.grad)
    print("Gradiente de tensor 3: ", c.grad)