
from abc import ABC, abstractmethod

from .neurons import Neuron

import numpy as np

class Layer(ABC):
    
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.trainable = True
    
    @abstractmethod
    def __call__(self, x:np.ndarray) -> np.ndarray:
        self.input = x
        pass

    @abstractmethod
    def backward(self, chained_grad) -> np.ndarray:
        pass

class Dense(Layer):
    
    def __init__(self, input_size:int, output_size:int, weight_init_method='he'):
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = []
        for _ in range(self.output_size):
            n = Neuron(input_size, output_size, weight_init_method=weight_init_method)
            self.neurons.append(n)
        
    def __call__(self, x:np.ndarray) -> np.ndarray:
        super().__call__(x)
        assert len(x) == self.input_size, f"Expected input size '{self.input_size}' but received '{len(x)}'"
        
        output = np.array([ neuron(x) for neuron in self.neurons ])
        assert len(output) == self.output_size, f"CODE ERROR, ouput does not have the correct size {len(output)} != {self.output_size}"
        
        return output
    
    def backward(self, chained_grad:np.ndarray) -> np.ndarray:
        assert len(chained_grad) == self.output_size
        return chained_grad*...
        
    
# class Conv2D(Layer):
    
#     def __init__(self, filters, kernel_size, strides=None, padding=None) -> None:
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.padding = padding
        
#     def __call__(self, x:np.ndarray) -> np.ndarray:
#         super().__call__(x)
#         ...
    
# class BatchNorm2d(Layer):
#     ...
    
# class MaxPool2D(Layer):
    
#     def __call__(self, x:np.ndarray) -> np.ndarray:
#         super().__call__(x)
#         ...

################### Acivation Functions ###################
class Relu(Layer):
    
    def __init__(self) -> None:
        super().__init__()
        self.trainable = False
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        super().__call__(x)
        return np.maximum(0, x)
    
    def backward(self, chained_grad):
        ...
        
# class Sigmoid(Layer):
    
#     def __call__(self, x:np.ndarray) -> np.ndarray:
#         super().__call__(x)
#         return (1/(1 + np.exp(-x)))
    
#     def backward(self, grad):
#         """ grad is the """
#         f = 1/(1 + np.exp(-self.input))
#         df = f * (1 - f)
        
#         return df
    
class Softmax(Layer):
    
    def __init__(self) -> None:
        super().__init__()
        self.trainable = False
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        super().__call__(x)        
        exp = np.exp(x)
    
        return exp / np.sum(exp)
    
    def backward(self, z, chained_grad):
        """Unvectorized computation of the gradient of softmax.
        z: (T, 1) column array of input values.
        Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
        is DjSi - the partial derivative of Si w.r.t. input j.
        """
        Sz = self(z)
        N = z.shape[0]
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                D[i, j] = Sz[i, 0] * (np.float32(i == j) - Sz[j, 0])
        return D
    
###########################################################