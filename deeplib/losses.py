
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

epsilon = 1e-7


class Loss(ABC):
    
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def __call__(self, y_pred:np.ndarray, y_true:np.ndarray) -> Any:
        pass


class BCELoss:
    
    def __init__(self) -> None:
        pass
    
    def __call__(self, y_pred:np.ndarray, y_true:np.ndarray) -> float:
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        term_0 = (1-y_true) * np.log(1-y_pred + epsilon)
        term_1 = y_true * np.log(y_pred + epsilon)
        
        return -np.mean(term_0+term_1, axis=0)
    
    
class CrossEntropyLoss:

    def __init__(self) -> None:
        pass
    
    def __call__(self, y_pred:np.ndarray, y_true:np.ndarray) -> np.ndarray:
        if len(y_pred.shape) > 1:
            total_loss = []
            for pred, true in zip(y_pred, y_true):
                loss = self.loss(pred, true)
                total_loss.append(loss)
        else:
            total_loss = [self.loss(y_pred, y_true)]
        
        total_loss = np.array(total_loss)
        
        return total_loss
         
    def loss(self, y_pred, y_true) -> float:
        """ 1D vectors """
        return -np.sum(y_true * np.log(y_pred + epsilon))

    def backward(self, ):
        ...

class SparseCrossEntropyLoss:

    def __init__(self) -> None:
        pass
    
    def __call__(self, y_pred:np.ndarray, y_true:np.ndarray) -> float:
        ...