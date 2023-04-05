import numpy as np


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Tensor:
    
    def __init__(self, data, _children=(), _operation='None', requires_grad=True) -> None:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        assert isinstance(data, np.ndarray), "data must be a list or numpy array"
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        # internal variables used for autograd graph construction
        self._backward = None
        self._children = _children
        self._operation = _operation # the operation that produced this node, for graphviz / debugging
        
        if requires_grad:
            self.grad = np.zeros(data.shape)
            self._backward = lambda: None
    
    
    def __add__(self, summand):
        summand = summand if isinstance(summand, Tensor) else Tensor(summand)
        out = Tensor(self.data + summand.data, (self, summand), '+', requires_grad=self.requires_grad)
        
        def _backward():
            self.grad += out.grad.sum() if len(self.data.shape) == 0 else out.grad
            summand.grad += out.grad.sum() if len(summand.data.shape) == 0 else out.grad
        
        out._backward = _backward 
        
        return out
        
    def __mul__(self, factor):
        factor = factor if isinstance(factor, Tensor) else Tensor(factor)
        out = Tensor(self.data * factor.data, (self, factor), '*', requires_grad=self.requires_grad)
        
        def _backward():
            self_grad = factor.data * out.grad
            self.grad += self_grad.sum() if len(self.data.shape) == 0 else self_grad
            
            factor_grad = self.data * out.grad
            factor.grad += factor_grad.sum() if len(factor.data.shape) == 0 else factor_grad
            
        out._backward = _backward 
        
        return out
    
    def __pow__(self, power):
        assert isinstance(power, (int, float)), f"Power of type '{type(power)}' not supported"
        out = Tensor(self.data**power, (self,), f'**{power}', requires_grad=self.requires_grad)
        
        def _backward():
            self_grad = (power * self.data**(power-1)) * out.grad
            self.grad += self_grad.sum() if len(self.data.shape) == 0 else self_grad
                       
        out._backward = _backward
        
        return out
    
    def __rpow__(self, power):
        assert isinstance(power, (int, float)), f"Power of type '{type(power)}' not supported"
        out = Tensor(np.power(power, self.data), (self,), f'{power}**', requires_grad=self.requires_grad)
        
        def _backward():
            self_grad += (power**self.data * np.log(power)) * out.grad
            self.grad += self_grad.sum() if len(self.data.shape) == 0 else self_grad
                       
        out._backward = _backward
        
        return out
    
    def __matmul__(self, tensor):
        tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
        out = Tensor(np.matmul(self.data, tensor.data), (self, tensor), 'x', requires_grad=self.requires_grad)
        
        def _backward():
            self_grad = tensor.data * out.grad
            self.grad += self_grad.sum() if len(self.data.shape) == 0 else self_grad
            
            tensor_grad = self.data * out.grad
            tensor.grad += tensor_grad.sum() if len(tensor.data.shape) == 0 else tensor_grad
                       
        out._backward = _backward
        
        return out
    
    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("Trying to call backward on Tensor with requires_grad=False")
        
        if self._backward is None:
            self._backward = lambda: None

        if grad is None:
            if self.data.size > 1:
                raise RuntimeError("grad must be specified for non-scalar tensors")
            else:
                grad = 1
        
        # topological order all of the children in the graph
        ordered_nodes = []
        visited_nodes = set()
        def visit_node(node):
            if node not in visited_nodes:
                visited_nodes.add(node)
                for child in node._children:
                    visit_node(child)
                ordered_nodes.append(node)
        visit_node(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = grad
        for v in reversed(ordered_nodes):
            v._backward()
    
    def __len__(self) -> int:
        return len(self.data)
    
    @property
    def shape(self) -> tuple:
        return self.data.shape
    
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
        
    def __repr__(self) -> str:
        return f"Tensor(value={self.data}, shape={self.shape}, op={self._operation})"
    
   
    @staticmethod
    def concat(tensors, dim=0):
        # Check that all tensors have the same shape along the specified dim
        dim_sizes = [tensor.shape[dim] for tensor in tensors]
        assert all(size == dim_sizes[0] for size in dim_sizes), f"Shapes along dim {dim} don't match: {[tensor.shape for tensor in tensors]}"

        # Create a list of slice objects to select the correct section of each tensor
        slices = []
        start = 0
        for tensor in tensors:
            slices.append(slice(start, start + tensor.shape[dim]))
            start += tensor.shape[dim]

        # Concatenate the sections along the specified dim
        if dim == 0:
            new_data = np.concatenate([tensor.data for tensor in tensors], axis=dim)
        else:
            new_data = np.concatenate([tensor.data.take(s, axis=dim) for tensor, s in zip(tensors, slices)], axis=dim)

        out = Tensor(new_data, tensors, _operation='concatenate', requires_grad=True)

        def _backward():
            # Split the gradient along the concatenated dim and backpropagate to each input tensor
            grads = np.split(out.grad, len(tensors), axis=dim)
            for tensor, grad in zip(tensors, grads):
                if dim == 0:
                    tensor.grad += grad
                else:
                    tensor.grad += grad.take(slices[tensors.index(tensor)], axis=dim)

        out._backward = _backward
        
        return out
    
    def view(self, shape:tuple):
        out = Tensor(self.data.reshape(shape), (self,), 'view', requires_grad=self.requires_grad)
        
        def _backward():
            self.grad += out.grad.reshape(self.shape)
            
        out._backward = _backward
        
        return out
    
    def unsqueeze(self, dim):
        data = np.expand_dims(self.data, dim)
        out = Tensor(data, (self,), 'unsqueeze', requires_grad=self.requires_grad)
        
        def _backward():
            self.grad += np.squeeze(out.grad, dim)
            
        out._backward = _backward
        
        return out
    
    def sum(self, dim=None):
        out = Tensor(self.data.sum(axis=dim), (self,), 'sum', requires_grad=self.requires_grad)
        
        def _backward():
            self.grad += np.ones(self.shape) * out.grad
            
        out._backward = _backward
        
        return out

# Check with pytorch that gradients are correct when applying different tensor operations
if __name__ == "__main__":
    l1 = [-4.0, 0, 5.0]
    l2 = [2.0, 2,  3.0]
    
    a = Tensor(l1, requires_grad=True).unsqueeze(0)
    b = Tensor(l2, requires_grad=True).unsqueeze(0)
    c = Tensor(4.0, requires_grad=True)
    out = Tensor.concat((a*c, b), dim=0)
    out = out.view((3,2))
    out.sum().backward()
    
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
    out = torch.concat((a*c, b), dim=0)
    out = out.view((3,2))

    # Calculamos el gradiente del tensor resultado
    out.sum().backward()

    # Mostramos los resultados
    print("\nTensor 1: ", a)
    print("Tensor 2: ", b)
    print("Tensor 3:", c)
    print("Tensor resultado: ", out)
    print("Gradiente de tensor 1: ", a.grad)
    print("Gradiente de tensor 2: ", b.grad)
    print("Gradiente de tensor 3: ", c.grad)
    
    