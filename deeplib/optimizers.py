
from abc import ABC, abstractmethod

from .modules import Model
from .modules import Layer

import numpy as np

class Optimizer(ABC):
    
    def __init__(self, model:Model, lr=0.001) -> None:
        super().__init__()
        self.model = model 
        self.lr = lr
        
    @abstractmethod
    def step(self, loss_gradient):
        pass


class GD(Optimizer):
    
    def step(self, loss_gradient):
        grads = loss_gradient
        for layer in self.model.layers[::-1]:
            # print("previous gradient", grads)
            # print(layer.__class__)
            # print("Gradients", grads)
            layer:Layer
            # Update weights
            if layer.trainable:
                for j, previous_layer_activation in enumerate(layer.input):
                    for i, (neuron, neuron_grad) in enumerate(zip(layer.neurons, grads[j])):
                        # Update bias
                        neuron.bias = neuron.bias - self.lr*neuron_grad
                        # Update weights
                        neuron_weight_gradients = previous_layer_activation * neuron_grad
                        neuron.weights = neuron.weights - self.lr*neuron_weight_gradients       
            # Propagate error
            grads = layer.backward(grads)
        
    
class SGD:
    ...
    
class Adam:
    ...