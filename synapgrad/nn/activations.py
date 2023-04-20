import numpy as np
from .. import Tensor, nn
from .functional import relu_fn, tanh_fn, sigmoid_fn, softmax_fn,  log_softmax_fn, epsilon


# ----------------------------- Modules -----------------------------
# -------------------------------------------------------------------
class ReLU(nn.Module):
    
    def forward(self, x:Tensor) -> Tensor:
        relu = relu_fn(x.data)
        out = Tensor(relu, (x,), '<ReLU>', requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                x._grad += (out.data > 0) * out._grad
        
        out._backward = _backward

        return out


class Tanh(nn.Module):
    """ np.sinh(x)/np.cosh(x) or -1j * np.tan(1j*x) """
    
    def forward(self, x:Tensor) -> Tensor:
        tanh = tanh_fn(x.data)
        out = Tensor(tanh, (x,), '<Tanh>', requires_grad=x.requires_grad)
    
        def _backward():
            if x.requires_grad:
                x_grad = 1 - tanh**2
                x._grad += x_grad * out._grad
            
        out._backward = _backward
        
        return out


class Sigmoid(nn.Module):
    
    def forward(self, x:Tensor) -> Tensor:
        sigmoid = sigmoid_fn(x.data)
        out = Tensor(sigmoid, (x,), '<Sigmoid>', requires_grad=x.requires_grad)
    
        def _backward():
            if x.requires_grad:
                x_grad = sigmoid * (1 - sigmoid)
                x._grad += x_grad * out._grad
            
        out._backward = _backward
        
        return out

    
class Softmax(nn.Module):
    """ 
    References: 
        https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/
    """
    
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        softmax = softmax_fn(x.data, self.dim)
        out = Tensor(softmax, (x,), '<Softmax>', requires_grad=x.requires_grad)
    
        def _backward():
            # Hand made derivation
            if x.requires_grad:
                jacobians = np.stack([np.diag(y) - np.outer(y, y) for y in softmax])
                out_grad = np.expand_dims(out._grad, axis=self.dim)
                x._grad += (out_grad @ jacobians).sum(axis=self.dim)
            
        out._backward = _backward
        
        return out

    
class LogSoftmax(nn.Module):
    """ 
    Same as Softmax(dim=self.dim)(x).log() but more numerically stable due to the log-sum-exp trick
            
    Reference to log-sum-exp trick: 
        https://en.wikipedia.org/wiki/LogSumExp 
    """
    
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        log_softmax = log_softmax_fn(x.data, self.dim)
        out = Tensor(log_softmax, (x,), '<LogSoftmax>', requires_grad=x.requires_grad)
    
        def _backward():
            if x.requires_grad:
                softmax = np.exp(log_softmax)
                jacobians = np.stack([np.diag(y) - np.outer(y, y) for y in softmax])
                dlog_dsoftmax = (1/(softmax + epsilon)) * out._grad
                dlog_dsoftmax = np.expand_dims(dlog_dsoftmax, axis=self.dim)
                x._grad += (dlog_dsoftmax @ jacobians).sum(axis=self.dim)
            
        out._backward = _backward
        
        return out
    