
from abc import ABC, abstractmethod

from .. import engine, Tensor
import numpy as np


class Optimizer(ABC):
    
    def __init__(self, parameters:list[Tensor], lr=0.001) -> None:
        super().__init__()
        self.parameters = parameters
        self.lr = lr
        self.t = 0
        
    def zero_grad(self):
        for p in self.parameters:
            p.zero_()
        
    @abstractmethod
    def step(self):
        self.t += 1


class SGD(Optimizer):
    """Reference: 
        https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    """
    
    def __init__(self, parameters: list[Tensor], lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, maximize=False) -> None:
        """
        Implements stochastic gradient descent (optionally with momentum).

        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            momentum (float, optional): Momentum factor (default: 0).
            dampening (float, optional): Dampening for momentum (default: 0).
            weight_decay (float, optional): Weight decay factor (default: 0).
            nesterov (bool, optional): Enables Nesterov momentum (default: False).
            maximize (bool, optional): Whether to maximize the objective (default: False).

        Returns:
            None
        """
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.momentum_buffer = []
        self.nesterov = nesterov
        self.dampening = dampening
        self.maximize = maximize
        self.weight_decay = weight_decay
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
    
    def step(self):
        super().step()
        with engine.no_grad():
            for i, p in enumerate(self.parameters):
                grad = p._grad
                
                # Weight decay
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay*p.data
                
                # Momentum
                if self.momentum != 0:
                    if self.t > 1:
                        self.momentum_buffer[i] = self.momentum*self.momentum_buffer[i] + (1 - self.dampening)*grad
                    else:
                        self.momentum_buffer.append(grad)
                
                    # Nesterov
                    if self.nesterov:
                        grad = grad + self.momentum*self.momentum_buffer[i]
                    else:
                        grad = self.momentum_buffer[i]
                
                # Update Parameter
                if self.maximize:
                    p.data += self.lr*grad
                else:
                    p.data -= self.lr*grad
        
    
class Adam(Optimizer):
    """Reference: 
        https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    """
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
        super().step()
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