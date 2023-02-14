
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
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = []
        for _ in range(self.output_size):
            n = Neuron(input_size, output_size, weight_init_method=weight_init_method)
            self.neurons.append(n)
        
    def __call__(self, x:np.ndarray) -> np.ndarray:
        super().__call__(x)
        assert x.shape[1] == self.input_size, f"Expected input size '{self.input_size}' but received '{x.shape[1]}'"
        
        output = np.array([ neuron(x) for neuron in self.neurons ]).transpose()
        assert len(output[0]) == self.output_size, f"CODE ERROR, ouput does not have the correct size {len(output[0])} != {self.output_size}"
        
        return output
    
    def backward(self, chained_grad:np.ndarray) -> np.ndarray:
        assert chained_grad.shape[1] == self.output_size
        weights = np.array([n.weights for n in self.neurons])
        # print("Weights", weights, weights.shape)
        # print("chained weights", chained_grad.shape)
        #print(chained_grad.shape, weights.shape)
        gradients = np.matmul(chained_grad, weights)
        
        return gradients # vector (1, input_size)
    
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

################### Activation Functions ###################
class Relu(Layer):
    
    def __init__(self) -> None:
        super().__init__()
        self.trainable = False
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        super().__call__(x)
        return np.maximum(0, x)
    
    def backward(self, chained_grad:np.ndarray) -> np.ndarray:
        gradient = np.where(self.input <= 0, 0, chained_grad)
        return gradient

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
        self.output = exp / np.sum(exp)
    
        return self.output
    
    def backward(self, chained_grad:np.ndarray) -> np.ndarray:
        """Unvectorized computation of the gradient of softmax.
        z: (T, 1) column array of input values.
        Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
        is DjSi - the partial derivative of Si w.r.t. input j.
        """
        jacobians = []
        for sample, output in zip(self.input, self.output):
            N = sample.shape[0]
            D = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    D[i, j] = output[i] * (np.float32(i == j) - output[j])
            jacobians.append(D)
        return np.array(jacobians)
    
###########################################################