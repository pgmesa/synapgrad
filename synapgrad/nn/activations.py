import numpy as np
from .. import Tensor, nn


# ---------------------------- Functions ----------------------------
# -------------------------------------------------------------------
def relu_fn(data:np.ndarray) -> np.ndarray:
    return np.maximum(0, data)


def sigmoid_fn(data:np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-data))


def softmax_fn(data:np.ndarray, dim:int) -> np.ndarray:
    # Shift to make it numerically stable (with large values 'inf' appears)
    shiftx = data - data.max(axis=dim, keepdims=True) 
    exps = np.exp(shiftx)
    exp_sums = exps.sum(axis=dim, keepdims=True)
    return exps / exp_sums

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
                x._grad += (x_grad) * out._grad
            
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
                diag = np.stack([np.diag(v) for v in softmax])
                S_matrix = np.stack([np.tile(v, softmax.shape[self.dim]) for v in softmax]).reshape(diag.shape)
                S_matrix_T = np.stack([np.transpose(m) for m in S_matrix])
                grad = diag - (S_matrix * S_matrix_T)
                x._grad += grad.sum(axis=self.dim) * out._grad
            
        out._backward = _backward
        
        return out

    
class LogSoftmax(nn.Module):
    
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        assert isinstance(x, Tensor), "Input must be a Tensor"
        softmax = softmax_fn(x.data, self.dim)
        log_softmax = np.log(softmax)
        out = Tensor(log_softmax, (x,), '<LogSoftmax>', requires_grad=x.requires_grad)
    
        def _backward():
            # Hand made derivation
            if x.requires_grad:
                I = np.stack((np.identity(softmax.shape[self.dim]),)*softmax.shape[0])
                S_matrix = np.stack([np.tile(v, softmax.shape[self.dim]) for v in softmax]).reshape(I.shape)
                S_matrix_T = np.stack([np.transpose(m) for m in S_matrix])
                grad = I - S_matrix_T
                x._grad += grad.sum(axis=self.dim) * out._grad
            
        out._backward = _backward
        
        return out
    