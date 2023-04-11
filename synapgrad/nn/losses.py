from typing import Any
from abc import ABC, abstractmethod
from .. import Tensor
from .activations import softmax_fn, LogSoftmax, relu_fn
import numpy as np


epsilon = 1e-7

# ---------------------------- Functions ----------------------------
# -------------------------------------------------------------------
def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return (y_pred - y_true)**2
    
    
def nll_loss(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return -y_pred[range(len(y_pred)), y_true].reshape((-1, 1))


def bce_loss(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    term_0 = (1-y_true) * np.log(1-y_pred + epsilon)
    term_1 = y_true * np.log(y_pred + epsilon)
    
    return -(term_0 + term_1)


def bce_with_logits_loss(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    tn = -relu_fn(y_pred)
    term = (1-y_true) * y_pred + tn + np.log(np.exp(-tn) + np.exp((-y_pred-tn)))
    
    return term


def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    softmax = np.log(softmax_fn(y_pred, 1))
    log_likelihood = nll_loss(softmax, y_true)
    return log_likelihood

# ----------------------------- Modules -----------------------------
# -------------------------------------------------------------------
class Loss(ABC):
    
    def __init__(self, reduction='mean') -> None:
        self.reduction = reduction
        
    def __call__(self, y_pred:Tensor, y_true:Tensor) -> Any:
        assert isinstance(y_pred, Tensor) and isinstance(y_true, Tensor), "Inputs must be Tensors"

        loss = self.criterion(y_pred, y_true)
        
        # Reduction
        if self.reduction == 'sum':
            reduction = loss.sum()
        elif self.reduction == 'mean':
            reduction = loss.mean()
        else:
            reduction = loss    
          
        if self.reduction is not None:
            reduction._operation = loss._operation.replace(">", reduction._operation.replace("<", ""))
            
        return reduction
    
    @abstractmethod
    def criterion(self, y_pred:Tensor, y_true:Tensor) -> Tensor:
        pass


class MSELoss(Loss):
    """ Mean Squared Error Loss: (y_pred - y_true)**2 """
    
    def criterion(self, y_pred:Tensor, y_true:Tensor) -> Tensor:
        assert y_pred.matches_shape(y_true), f"Inputs shape don't match y_pred={y_pred.shape}, y_true={y_true.shape}"
        req_grad = y_pred.requires_grad or y_true.requires_grad
        mse = mse_loss(y_pred.data, y_true.data)
        loss = Tensor(mse, (y_pred, y_true), '<MSELoss>', requires_grad=req_grad)
        
        def _backward():
            if y_pred.requires_grad: 
                grad = 2*(y_pred.data - y_true.data) 
                y_pred._grad += grad * loss._grad
        
        loss._backward = _backward
        
        return loss


class NLLLoss(Loss):
    """ This class expects y_true to be the probabilities of a LogSoftmax function. Also, y_pred 
        should be shape=(batch, num_classes). It expects to receive the probability of a sample to be of each class.
        For binary classification problems, output should not be a scalar value (0 <= x <=1) but an array with 
        the probability of each class [0.3, 0.7]
        reference: https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81    
    """
    
    def criterion(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        assert isinstance(y_pred, Tensor) and isinstance(y_true, Tensor), "Inputs must be Tensors"
        req_grad = y_pred.requires_grad or y_true.requires_grad
        log_likelihood = nll_loss(y_pred.data, y_true.data)
        loss = Tensor(log_likelihood, (y_pred, y_true), '<NLLLoss>', requires_grad=req_grad)
        
        def _backward():
            # Hand made derivation
            if y_pred.requires_grad:
                grad = np.zeros(y_pred.shape)
                for i, true in enumerate(y_true.data):
                    grad[i][true] = -1.0
                y_pred._grad += (grad * loss._grad)
        
        loss._backward = _backward
        
        return loss

   
class BCELoss(Loss):
    
    def criterion(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        req_grad = y_pred.requires_grad or y_true.requires_grad
        bce = bce_loss(y_pred.data, y_true.data)
        loss = Tensor(bce, (y_pred, y_true), '<BCELoss>', requires_grad=req_grad)
        
        def _backward():
            # Hand made derivation
            if y_pred.requires_grad:
                term_0 = -(1 - y_true.data) / ((1 - y_pred.data) + epsilon)
                term_1 = y_true.data / (y_pred.data + epsilon)
                y_pred._grad += -(term_0 + term_1) * loss._grad
        
        loss._backward = _backward
        
        return loss


class BCEWithLogitsLoss(Loss):
    """ This loss combines a Sigmoid layer and the BCELoss in one single class.
    This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as,
    by combining the operations into one layer, we take advantage of the log-sum-exp
    trick for numerical stability
    
    References:
        https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        https://stackoverflow.com/questions/66906884/how-is-pytorchs-class-bcewithlogitsloss-exactly-implemented
    """
    
    def criterion(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        req_grad = y_pred.requires_grad or y_true.requires_grad
        bce = bce_with_logits_loss(y_pred.data, y_true.data)
        loss = Tensor(bce, (y_pred, y_true), '<BCEWithLogitsLoss>', requires_grad=req_grad)
        
        def _backward():
            # Hand made derivation
            if y_pred.requires_grad:
                tn = -relu_fn(y_pred.data)
                dtn = np.where(tn == 0, 0, -1)
                div1 = -dtn*np.exp(-tn) + (-1-dtn)*np.exp((-y_pred.data-tn))
                div2 = np.exp(-tn) + np.exp((-y_pred.data-tn))
                grad = (1 - y_true.data) + dtn + (div1/(div2 + epsilon))
                
                y_pred._grad += grad * loss._grad
        
        loss._backward = _backward
        
        return loss
    
    
class CrossEntropyLoss(Loss):
    """ Same as:\n
    log_softmax = LogSoftmax(dim=1)(y_pred)\n
    loss = NLLLoss(reduction=None)(log_softmax, y_true)
    
    Reference: https://deepnotes.io/softmax-crossentropy#:~:text=Cross%20entropy%20indicates%20the%20distance,used%20alternative%20of%20squared%20error
    """
    
    def criterion(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        y_pred is the output (logits) from fully connected layer (num_examples x num_classes)
        y_true is labels (num_examples x 1)
            Note that y is not one-hot encoded vector. 
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """        
        req_grad = y_pred.requires_grad or y_true.requires_grad
        cross_entropy = cross_entropy_loss(y_pred.data, y_true.data)
        loss = Tensor(cross_entropy, (y_pred, y_true), '<CrossEntropyLoss>', requires_grad=req_grad)
        
        def _backward():
            # Hand made derivation
            if y_pred.requires_grad:
                dlogits = softmax_fn(y_pred.data, 1)
                n = y_pred.shape[0]
                dlogits[range(n), y_true.data] -= 1
                y_pred._grad += (dlogits * loss._grad)
        
        loss._backward = _backward
        
        return loss
        