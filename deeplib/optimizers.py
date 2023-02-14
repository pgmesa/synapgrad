
from abc import ABC, abstractmethod

from .models import Model
from .models import Layer


class Optimizer(ABC):
    

    def __init__(self, model:Model) -> None:
        super().__init__()
        self.model = model
        
    @abstractmethod
    def step(self, loss_gradient):
        pass


class GD(Optimizer):
    
    def step(self, loss_gradient):
        grads = loss_gradient
        for layer in self.model.layers[::-1]:
            layer:Layer
            grads = layer.backward(grads)
            # TODO: Actualizar pesos
        
    
class SGD:
    ...
    
class Adam:
    ...