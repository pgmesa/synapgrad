from typing import Any
from abc import ABC, abstractmethod
from .. import Tensor
import numpy as np


epsilon = 1e-7


class Loss(ABC):
    
    def __init__(self, reduction='mean') -> None:
        self.reduction = reduction
        
    def __call__(self, y_pred:Tensor, y_true:Tensor) -> Any:
        assert y_pred.matches_shape(y_true), f"Inputs shape don't match y_pred={y_pred.shape}, y_true={y_true.shape}"

        loss = self.criterion(y_pred, y_true)
        
        # Reduction
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
            
        return loss
    
    @abstractmethod
    def criterion(self, y_pred:Tensor, y_true:Tensor) -> Tensor:
        pass


class MSELoss(Loss):
    """ Mean Squared Error Loss: (y_pred - y_true)**2 """
    
    def criterion(self, y_pred:Tensor, y_true:Tensor) -> Tensor:
        assert isinstance(y_pred, Tensor) and isinstance(y_true, Tensor), "Inputs must be Tensors"
        req_grad = y_pred.requires_grad or y_true.requires_grad
        loss = Tensor((y_pred.data - y_true.data)**2, (y_pred, y_true), '<MSELoss>', requires_grad=req_grad)
        
        def _backward():
            grad = 2*(y_pred.data - y_true.data) * loss._grad
            if y_pred.requires_grad: y_pred._grad += grad # * loss._grad
            if y_true.requires_grad: y_true._grad += grad # * loss._grad
        
        loss._backward = _backward
        
        return loss
    

# class BCELoss(Loss):
#     # TODO: Not working properly
    
#     def __init__(self) -> None:
#         pass
    
#     def __call__(self, y_pred:np.ndarray, y_true:np.ndarray) -> float:
#         super().__call__(y_pred, y_true)
#         y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
#         term_0 = (1-y_true) * np.log(1-y_pred + epsilon)
#         term_1 = y_true * np.log(y_pred + epsilon)
        
#         return -np.mean(term_0+term_1, axis=0)
    
    
class CrossEntropyLoss(Loss):
    """ Reference: https://deepnotes.io/softmax-crossentropy#:~:text=Cross%20entropy%20indicates%20the%20distance,used%20alternative%20of%20squared%20error"""
    # TODO: Not working properly
    
    def criterion(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
            Note that y is not one-hot encoded vector. 
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        assert isinstance(y_pred, Tensor) and isinstance(y_true, Tensor), "Inputs must be Tensors"
        return -y_pred[range(y_true.shape[0]), y_true.data.argmax(axis=1)].log().mean()
                    
        