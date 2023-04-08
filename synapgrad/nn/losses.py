from typing import Any
from abc import ABC, abstractmethod
from .. import Tensor


epsilon = 1e-7


class Loss(ABC):
    
    def __init__(self, reduction='mean') -> None:
        self.reduction = reduction
        
    def __call__(self, y_pred:Tensor, y_true:Tensor) -> Any:
        assert y_pred.matches_shape(y_true), f"Inputs shape don't match y_pred={y_pred.shape}, y_true={y_true.shape}"
        
        if len(y_pred.shape) > 1:
            raise ValueError("Module expects a batched input of Shape=(batch_size,)")

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
        loss = Tensor((y_pred.data - y_true.data)**2, (y_pred, y_true), '<MeanSquaredError>', requires_grad=req_grad)
        
        def _backward():
            grad = 2*(y_pred.data - y_true.data) * loss._grad
            if y_pred.requires_grad: y_pred._grad += grad
            if y_true.requires_grad: y_true._grad += grad
        
        loss._backward = _backward
        
        return loss
    

# class BCELoss(Loss):
    
#     def __init__(self) -> None:
#         pass
    
#     def __call__(self, y_pred:np.ndarray, y_true:np.ndarray) -> float:
#         super().__call__(y_pred, y_true)
#         y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
#         term_0 = (1-y_true) * np.log(1-y_pred + epsilon)
#         term_1 = y_true * np.log(y_pred + epsilon)
        
#         return -np.mean(term_0+term_1, axis=0)
    
    
# class CrossEntropyLoss:

#     def __init__(self) -> None:
#         pass
    
#     def __call__(self, y_pred:np.ndarray, y_true:np.ndarray) -> np.ndarray:
#         super().__call__(y_pred, y_true)
#         total_loss = []
#         for pred, true in zip(y_pred, y_true):
#             loss = -np.sum(true * np.log(pred + epsilon))
#             total_loss.append(loss)
        
#         total_loss = np.array(total_loss)
        
#         return total_loss

#     def backward(self):
#         """Gradient of the cross-entropy loss function for p and y.
#         p: (T, 1) vector of predicted probabilities.
#         y: (T, 1) vector of expected probabilities; must be one-hot -- one and only
#                 one element of y is 1; the rest are 0.
#         Returns a (1, T) Jacobian for this function.
#         """
#         #print(self.y_pred, self.y_pred.shape, self.y_true, self.y_true.shape)
#         loss_gradient = []
#         for pred, true in zip(self.y_pred, self.y_true):
#             pred = np.expand_dims(pred, axis=-1)
#             true = np.expand_dims(true, axis=-1)
#             assert(pred.shape == true.shape and pred.shape[1] == 1)
#             # py is the value of p at the index where y == 1 (one and only one such
#             # index is expected for a one-hot y).
#             py = pred[true == 1]
#             #print(py)
#             assert(py.size == 1)
#             # D is zeros everywhere except at the index where y == 1. The final D has
#             # to be a row-vector.
#             D = np.zeros_like(pred)
#             D[pred == 1] = -1/py.flat[0]
#             loss_gradient.append(D.flatten())
        
#         loss_gradient = np.array(loss_gradient)
#         print(loss_gradient, loss_gradient.shape)
#         return loss_gradient