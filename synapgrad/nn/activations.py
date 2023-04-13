import numpy as np
from .. import Tensor, nn


# ---------------------------- Functions ----------------------------
# -------------------------------------------------------------------
def relu_fn(data:np.ndarray) -> np.ndarray:
    return np.maximum(0, data)


def tanh_fn(data:np.ndarray) -> np.ndarray:
    return np.tanh(data)


def sigmoid_fn(data:np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-data))


def softmax_fn(data:np.ndarray, dim:int) -> np.ndarray:
    # Shift to make it numerically stable (with large values 'inf' appears)
    shiftx = data - data.max(axis=dim, keepdims=True) 
    exps = np.exp(shiftx)
    exp_sums = exps.sum(axis=dim, keepdims=True)
    return exps / exp_sums

def log_softmax_fn(data:np.ndarray, dim:int) -> np.ndarray:
    # Using log-sum-exp trick for numerical stability
    max_val = data.max(axis=dim, keepdims=True)
    substract = data - max_val
    exp = np.exp(substract)
    lse = max_val + np.log(exp.sum(axis=dim, keepdims=True))
    log_softmax = data - lse
    return log_softmax

# ----------------------------- Modules -----------------------------
# -------------------------------------------------------------------
class ReLU(nn.Module):
    
    def forward(self, x:Tensor) -> Tensor:
        assert isinstance(x, Tensor), "Input must be a Tensor" 
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
        assert isinstance(x, Tensor), "Input must be a Tensor"
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
        # Returning (1/(1 + np.e**-x)) should be enough, but defining explicit 
        # grad function should be faster
        assert isinstance(x, Tensor), "Input must be a Tensor"
        sigmoid = sigmoid_fn(x.data)
        out = Tensor(sigmoid, (x,), '<Sigmoid>', requires_grad=x.requires_grad)
    
        def _backward():
            if x.requires_grad:
                x_grad = sigmoid * (1 - sigmoid)
                x._grad += x_grad * out._grad
            
        out._backward = _backward
        
        return out

    
class Softmax(nn.Module):
    """ References: 
            https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
            https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/
    """
    
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        assert isinstance(x, Tensor), "Input must be a Tensor"
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
    """ Same as Softmax(dim=self.dim)(x).log() but more numerically stable due to the log-sum-exp trick
            Reference to log-sum-exp trick: https://en.wikipedia.org/wiki/LogSumExp 
    """
    
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        assert isinstance(x, Tensor), "Input must be a Tensor"
        log_softmax = log_softmax_fn(x.data, self.dim)
        out = Tensor(log_softmax, (x,), '<LogSoftmax>', requires_grad=x.requires_grad)
    
        def _backward():
            if x.requires_grad:
                softmax = np.exp(log_softmax)
                jacobians = np.stack([np.diag(y) - np.outer(y, y) for y in softmax])
                dlog_dsoftmax = (1/softmax) * out._grad
                dlog_dsoftmax = np.expand_dims(dlog_dsoftmax, axis=self.dim)
                x._grad += (dlog_dsoftmax @ jacobians).sum(axis=self.dim)
            
        out._backward = _backward
        
        return out
    