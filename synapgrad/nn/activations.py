from synapgrad.tensor import Tensor
from synapgrad import nn
from synapgrad.nn import functional as F


# ************************************
# ******* Activation Modules *********
# ************************************

class ReLU(nn.Module):
    """
    ReLU activation function. 
    
    The ReLU activation function is defined as:
    f(x) = max(0, x)
    """
    
    def forward(self, x:Tensor) -> Tensor:
        return F.relu(x)
    
    
class LeakyReLU(nn.Module):
    """
    LeakyReLU activation function. 
    
    The LeakyReLU activation function is defined as:
    f(x) = max(0, x) + negative_slope * min(0, x)
    """
    
    def __init__(self, negative_slope:float=0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        
    def forward(self, x:Tensor) -> Tensor:
        return F.leaky_relu(x, self.negative_slope)
    
    
class SELU(nn.Module):
    """
    SELU activation function.
    
    The SELU activation function is defined as:
    f(x) = scale * (max(0, x) + min(0, alpha*(exp(x)-1))
    
    with:
        - alpha = 1.6732632423543772848170429916717
        - scale = 1.0507009873554804934193349852946
    """
    
    def forward(self, x:Tensor) -> Tensor:
        return F.selu(x)
    

class Tanh(nn.Module):
    """ 
    Tanh activation function
    
    It is defined as np.sinh(x)/np.cosh(x) or -1j * np.tan(1j*x) 
    """
    
    def forward(self, x:Tensor) -> Tensor:
        return F.tanh(x)


class Sigmoid(nn.Module):
    """ 
    Sigmoid activation function
    
    It is defined as 1/(1+np.exp(-x))
    """
    
    def forward(self, x:Tensor) -> Tensor:
        return F.sigmoid(x)

    
class Softmax(nn.Module):
    """ 
    Softmax activation function
    
    It is defined as exp(x)/sum(exp(x))
    """
    
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x, self.dim)

    
class LogSoftmax(nn.Module):
    """ 
    Same as Softmax(dim=self.dim)(x).log() but more numerically stable due to the log-sum-exp trick
    
    It is defined as log(softmax(x))
            
    Reference to log-sum-exp trick: 
    - https://en.wikipedia.org/wiki/LogSumExp 
    """
    
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        return F.log_softmax(x, self.dim)
    