
from abc import ABC, abstractmethod

from .neurons import Neuron

import numpy as np

class Layer(ABC):
    
    @abstractmethod
    def __call__(self, x:np.ndarray) -> np.ndarray:
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
    
class Conv2D(Layer):
    
    def __init__(self, filters, kernel_size, strides=None, padding=None) -> None:
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        
    def __call__(self, x:np.ndarray) -> np.ndarray:
        super().__call__(x)
        ...
    
class BatchNorm2d(Layer):
    ...
    
class MaxPool2D(Layer):
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        super().__call__(x)
        ...

################### Acivation Functions ###################
class Relu(Layer):
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        super().__call__(x)
        return np.maximum(0, x)

class Sigmoid(Layer):
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        super().__call__(x)
        return (1/(1 + np.exp(-x)))

    
class Softmax(Layer):
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        super().__call__(x)        
        exp = np.exp(x)
    
        return exp / np.sum(exp)
    
###########################################################