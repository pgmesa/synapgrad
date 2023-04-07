
from abc import ABC, abstractmethod

from . import engine
from .engine import Tensor


class Optimizer(ABC):
    
    def __init__(self, parameters:list[Tensor], lr=0.001) -> None:
        super().__init__()
        self.parameters = parameters
        self.lr = lr
        
    def zero_grad(self):
        for p in self.parameters:
            p.zero_()
        
    @abstractmethod
    def step(self):
        pass


class GD(Optimizer):
    
    def step(self):
        with engine.no_grad():
            for p in self.parameters:
                p -= self.lr*p.grad
        
    
class SGD(Optimizer):
    
    def step(self):
        with engine.no_grad():
            for p in self.parameters:
                ...
    
    
class Adam(Optimizer):
    
    def step(self):
        with engine.no_grad():
            for p in self.parameters:
                ...