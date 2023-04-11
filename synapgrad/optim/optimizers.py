
from abc import ABC, abstractmethod

from .. import engine, Tensor
import numpy as np


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


class SGD(Optimizer):
    
    def step(self):
        with engine.no_grad():
            for p in self.parameters:
                p.data -= self.lr*p._grad
        
    
class Adam(Optimizer):
    # TODO: Nos working properly
    
    def __init__(self, parameters: list[Tensor], lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8) -> None:
        """Simplified Adam optimizer -> https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
            amsgrad = False, maximize = False, weight_decay = 0
        Args:
            beta1 (float): Exponential decay rate for the moving average of the gradient.
            beta2 (float): Exponential decay rate for the moving average of the squared gradient.
            epsilon (float): Small value for numerical stability.
        """
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.mt = [0 for _ in range(len(parameters))]
        self.vt = [0 for _ in range(len(parameters))]
    
    def step(self):
        
        with engine.no_grad():
            for i, p in enumerate(self.parameters):
                # Update the moving average of the gradient
                m = self.beta1 * self.mt[i] + (1 - self.beta1) * p._grad
                self.mt[i] = m
                m_corrected = m / (1 - self.beta1)

                # Update the moving average of the squared gradient
                v = self.beta2 * self.vt[i]+ (1 - self.beta2) * p._grad**2
                self.vt[i] = v
                v_corrected = v / (1 - self.beta2)

                # Update the parameters using the Adam formula
                p.data -= self.lr * m_corrected / (np.sqrt(v_corrected) + self.epsilon)