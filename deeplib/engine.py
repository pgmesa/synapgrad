from abc import abstractclassmethod, ABC
import numpy as np


class Module(ABC):

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    @abstractclassmethod
    def parameters(self):
        return []


class Tensor:
    
    def __init__(self, data, _children=(), _operation='') -> None:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        assert isinstance(data, np.ndarray), "data must be a list or numpy array"
        self.data = data
        self.grad = np.zeros(data.shape)
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._children = _children
        self._operation = _operation # the operation that produced this node, for graphviz / debugging
    
    
    def __add__(self, summand):
        summand = summand if isinstance(summand, Tensor) else Tensor(summand)
        out = Tensor(self.data + summand.data, (self, summand), '+')
        
        def _backward():
            self.grad += out.grad
            summand.grad += out.grad
        
        out._backward = _backward 
        
        return out
        
    def __mul__(self, factor):
        factor = factor if isinstance(factor, Tensor) else Tensor(factor)
        out = Tensor(self.data * factor.data, (self, factor), '*')
        
        def _backward():
            self.grad += factor.data * out.grad
            factor.grad += self.data * out.grad
            
        out._backward = _backward 
        
        return out
    
    def __pow__(self, power):
        assert isinstance(power, (int, float)), f"Power of type '{type(power)}' not supported"
        out = Tensor(self.data**power, (self,), f'**{power}')
        
        def _backward():
            self.grad += (power * self.data**(power-1)) * out.grad
                       
        out._backward = _backward
        
        return out
    
    def __rpow__(self, power):
        assert isinstance(power, (int, float)), f"Power of type '{type(power)}' not supported"
        out = Tensor(np.power(power, self.data), (self,), f'{power}**')
        
        def _backward():
            self.grad += (power**self.data * np.log(power)) * out.grad
                       
        out._backward = _backward
        
        return out
    
    def __matmul__(self, tensor):
        tensor = tensor if isinstance(tensor, Tensor) else Tensor(tensor)
        out = Tensor(np.matmul(self.data, tensor.data), (self, tensor), 'x')
        
        def _backward():
            self.grad += tensor.data * out.grad
            tensor.grad += self.data * out.grad
                       
        out._backward = _backward
        
        return out
    
    def backward(self):
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
        self.grad = 1
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
        return f"Tensor(value={self.data}, grad={self.grad})"
        
    
if __name__ == "__main__":
    l1 = [-4.0, 5.0]
    l2 = [2.0, 3.0]
    
    a = Tensor(l1)
    b = Tensor(l2)
    c = 2 ** a 
    c.backward()
    
    print("Tensor 1: ", a)
    print("Tensor 2: ", b)
    print("Tensor resultado: ", c)
    print("Gradiente de tensor 1: ", a.grad)
    print("Gradiente de tensor 2: ", b.grad)
    
    import torch
    from torch import nn
    # Creamos el tensor con valores diferentes
    tensor1 = torch.tensor(l1, requires_grad=True)
    tensor2 = torch.tensor(l2, requires_grad=True)

    # Realizamos la multiplicación entre los dos tensores
    tensor_resultado = 2 ** tensor1 

    # Calculamos el gradiente del tensor resultado
    if len(tensor_resultado.shape) >= 1:
        tensor_resultado.backward(torch.ones(tensor_resultado.shape))
    else:
        tensor_resultado.backward()

    # Mostramos los resultados
    print("Tensor 1: ", tensor1)
    print("Tensor 2: ", tensor2)
    print("Tensor resultado: ", tensor_resultado)
    print("Gradiente de tensor 1: ", tensor1.grad)
    print("Gradiente de tensor 2: ", tensor2.grad)
    
    