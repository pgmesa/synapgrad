
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

epsilon = 1e-7


class Loss(ABC):
    
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def __call__(self, y_pred:np.ndarray, y_true:np.ndarray) -> Any:
        self.y_pred = y_pred
        self.y_true = y_true
        
    @abstractmethod
    def backward(self):
        pass


class MeanAbsoluteError(Loss):
    
    def __init__(self) -> None:
        pass
    
    def __call__(self, y_pred:np.ndarray, y_true:np.ndarray) -> Any:
        super().__call__(y_pred, y_true)
        return (y_pred - y_true)**2
    
    def backward(self):
        return 2*(self.y_pred - self.y_true)


class BCELoss(Loss):
    
    def __init__(self) -> None:
        pass
    
    def __call__(self, y_pred:np.ndarray, y_true:np.ndarray) -> float:
        super().__call__(y_pred, y_true)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        term_0 = (1-y_true) * np.log(1-y_pred + epsilon)
        term_1 = y_true * np.log(y_pred + epsilon)
        
        return -np.mean(term_0+term_1, axis=0)
    
    
class CrossEntropyLoss(Loss):

    def __init__(self) -> None:
        pass
    
    def __call__(self, y_pred:np.ndarray, y_true:np.ndarray) -> np.ndarray:
        super().__call__(y_pred, y_true)
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

    def backward(self):
        """Gradient of the cross-entropy loss function for p and y.
        p: (T, 1) vector of predicted probabilities.
        y: (T, 1) vector of expected probabilities; must be one-hot -- one and only
                one element of y is 1; the rest are 0.
        Returns a (1, T) Jacobian for this function.
        """
        assert(self.y_pred.shape == self.y_true.shape and self.y_pred.shape[1] == 1)
        # py is the value of p at the index where y == 1 (one and only one such
        # index is expected for a one-hot y).
        py = self.y_pred[self.y_true == 1]
        assert(py.size == 1)
        # D is zeros everywhere except at the index where y == 1. The final D has
        # to be a row-vector.
        D = np.zeros_like(self.y_pred)
        D[self.y_pred == 1] = -1/py.flat[0]
        return D.flatten()