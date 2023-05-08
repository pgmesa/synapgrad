
from abc import ABC, abstractmethod

from .. import Tensor, tensor
import numpy as np


class Optimizer(ABC):
    
    def __init__(self, parameters:list[Tensor], lr=0.001) -> None:
        super().__init__()
        if len(parameters) == 0:
            raise ValueError("optimizer got an empty parameter list")
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
    """
    Reference: 
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
        with tensor.no_grad():
            for i, p in enumerate(self.parameters):
                grad = p._grad
                
                # Weight decay
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay*p.data
                
                # Momentum
                if self.momentum != 0:
                    if self.t > 1:
                        self.momentum_buffer[i] = self.momentum*self.momentum_buffer[i] + (1.0 - self.dampening)*grad
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
    """
    Reference: 
        https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    """
    
    def __init__(self, parameters: list[Tensor], lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                    weight_decay=0, maximize=False) -> None:
        """
        Implements Adam algorithm.
        
        Args:
            params (iterable): Iterable of model parameters to optimize.
            lr (float, optional): Learning rate. Default is 0.001.
            betas (tuple, optional): Coefficients used for computing running averages of gradient and its square.
                                    Default is (0.9, 0.999).
            eps (float, optional): Term added to the denominator to improve numerical stability. Default is 1e-08.
            weight_decay (float, optional): Weight decay (L2 penalty) factor. Default is 0.
            maximize (bool, optional): Whether to maximize or minimize the objective function. Default is False.
        
        """
        super().__init__(parameters, lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.epsilon = eps
        self.weight_decay = weight_decay
        self.maximize = maximize
        
        self.m1 = [0 for _ in range(len(parameters))]
        self.m2 = [0 for _ in range(len(parameters))]
    
    def step(self):
        super().step()
        with tensor.no_grad():
            for i, p in enumerate(self.parameters):
                grad = -p._grad if self.maximize else p._grad   
                    
                # Weight decay
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay*p.data
                    
                # Update biased first moment estimate
                self.m1[i] = self.beta1 * self.m1[i] + (1.0 - self.beta1) * grad
                
                # Update biased second raw moment estimate
                self.m2[i] = self.beta2 * self.m2[i] + (1.0 - self.beta2) * grad**2.0
                
                m1_corrected = self.m1[i] / (1.0 - self.beta1**self.t)
                m2_corrected = self.m2[i] / (1.0 - self.beta2**self.t)

                # Update the parameters using the Adam formula
                p.data -= (self.lr * m1_corrected) / (np.sqrt(m2_corrected) + self.epsilon)