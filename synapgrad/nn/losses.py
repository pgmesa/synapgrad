from typing import Any

from synapgrad.nn import Module
from synapgrad.tensor import Tensor
from synapgrad.nn import functional as F


# ******************************
# ******* Loss Modules *********
# ******************************

class Loss(Module):
    """ 
    Generic class for loss functions.
    """
    
    def __init__(self, reduction='mean') -> None:
        super().__init__()
        self.reduction = reduction
        
    def __call__(self, y_pred:Tensor, y_true:Tensor) -> Any:
        loss = super().__call__(y_pred, y_true)
                
        # Reduction
        if self.reduction == 'sum':
            reduction = loss.sum()
        elif self.reduction == 'mean':
            reduction = loss.mean()
        else:
            reduction = loss 
            
        return reduction


class MSELoss(Loss):
    """ 
    Mean Squared Error Loss: (y_pred - y_true) ** 2. It is mostly used for regression problems.
    
    Inputs:
    - y_pred: output (logits) from fully connected layer (num_examples x num_classes)
    - y_true: labels (num_examples x num_classes)
    """
    
    def forward(self, y_pred:Tensor, y_true:Tensor) -> Tensor:
        return F.mse_loss(y_pred, y_true)


class NLLLoss(Loss):
    """ 
    Negative Log Likelihood Loss
    
    Inputs:
    - y_pred: log of the class probabilities returned by LogSoftmax function. shape=(num_examples, num_classes).
        For binary classification problems, it should not be a scalar value (0 <= x <=1) but an array with 
        the probability of each class [-0.523, -0.155]
    - y_true: labels. shape=(num_examples,). Note that y is not a one-hot encoded vector but an array with each value in 
        the range [0, num_classes-1]
    
    Reference: 
    - https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81    
    """
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return F.nll_loss(y_pred, y_true)

   
class BCELoss(Loss):
    """
    Binary Cross Entropy loss function.

    Inputs:
    - y_pred: probabilities returned by Sigmoid function. Values in range [0, 1], shape=(num_examples,)
    - y_true: binary labels. 0 or 1, shape=(num_examples,)
    
    References:
    - https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    """
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return F.binary_cross_entropy(y_pred, y_true)


class BCEWithLogitsLoss(Loss):
    """ 
    Binary Cross Entropy with Logits loss function.
    
    This loss combines a Sigmoid layer and the BCELoss in one single class.
    This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as,
    by combining the operations into one layer, we take advantage of the log-sum-exp
    trick for numerical stability
    
    Inputs:
    - y_pred: output (logits) from fully connected layer. shape=(num_examples,)
    - y_true: binary labels. 0 or 1, shape=(num_examples,)
    
    References:
    - https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    - https://stackoverflow.com/questions/66906884/how-is-pytorchs-class-bcewithlogitsloss-exactly-implemented
    """
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(y_pred, y_true)
    
    
class CrossEntropyLoss(Loss):
    """ 
    Cross Entropy Loss function.
    
    Same as:
    - log_probs = LogSoftmax(dim=1)(y_pred)
    - loss = NLLLoss(reduction=None)(log_probs, y_true)
    
    Inputs:
    - y_pred: output (logits) from fully connected layer. shape=(num_examples, num_classes)
    - y_true: labels. shape=(num_examples,). Note that y is not a one-hot encoded vector but an array with each value in 
        the range [0, num_classes-1]
    
    References: 
    - https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    - https://deepnotes.io/softmax-crossentropy#:~:text=Cross%20entropy%20indicates%20the%20distance,used%20alternative%20of%20squared%20error
    """
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return F.cross_entropy(y_pred, y_true)
        