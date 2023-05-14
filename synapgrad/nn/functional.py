from synapgrad import cpu_ops
from synapgrad.tensor import Tensor
from synapgrad.autograd import Function, Context
from synapgrad.device import Device


# ************************************
# ******* Activation functions *******
# ************************************

class ReLU(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x1 to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.relu_forward(x.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device)

        ctx.save_for_backward(x)
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        x, = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.relu_backward(grad_output.data, x.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad


def relu(x:Tensor):
    """ 
    ReLU activation function. 
    
    The ReLU activation function is defined as:
    f(x) = max(0, x)

    Args:
        x (Tensor): tensor

    Returns:
        Tensor: result
    """
    return ReLU.apply(x)


class Tanh(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x1 to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.tanh_forward(x.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device)

        ctx.save_for_backward(out)
        
        return out
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        out, = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.tanh_backward(grad_output.data, out.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad
    

def tanh(x:Tensor):
    """ 
    Tanh activation function.

    Args:
        x (Tensor): tensor

    Returns:
        Tensor: result
    """
    return Tanh.apply(x)


class Sigmoid(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x1 to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.sigmoid_forward(x.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device)

        ctx.save_for_backward(out)
        
        return out
        
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        out, = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.sigmoid_backward(grad_output.data, out.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad
    
    
def sigmoid(x:Tensor):
    """ 
    Sigmoid activation function. 
    
    The Sigmoid activation function is defined as:
    f(x) = 1 / (1 + exp(-x))

    Args:
        x (Tensor): tensor

    Returns:
        Tensor: result
    """
    return Sigmoid.apply(x)


class Softmax(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, dim:int):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x1 to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.softmax_forward(x.data, dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device)

        ctx.save_for_backward(out)
        ctx.dim = dim
        
        return out
        
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        out, = ctx.saved_tensors
        dim = ctx.dim
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.softmax_backward(grad_output.data, out.data, dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad
    
    
def softmax(x:Tensor, dim:int):
    """ 
    Softmax activation function. 
    
    The Softmax activation function is defined as:
    f(x) = exp(x) / sum(exp(x))

    Args:
        x (Tensor): tensor

    Returns:
        Tensor: result
    """
    return Softmax.apply(x, dim)


class LogSoftmax(Function):
    
    @staticmethod
    def forward(ctx:Context, x:Tensor, dim:int):
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected x1 to be a Tensor but got {type(x)}")
        
        if x.device == Device.CPU:
            out_data = cpu_ops.log_softmax_forward(x.data, dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {x.device} not supported")

        out = Tensor(out_data, device=x.device)

        ctx.save_for_backward(out)
        ctx.dim = dim
        
        return out
        
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        out, = ctx.saved_tensors
        dim = ctx.dim
        
        if grad_output.device == Device.CPU:
            a_grad = cpu_ops.log_softmax_backward(grad_output.data, out.data, dim)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        x_grad = Tensor(a_grad, device=grad_output.device)
        
        return x_grad


def log_softmax(x:Tensor, dim:int):
    """ 
    LogSoftmax activation function. 
    
    The LogSoftmax activation function is defined as:
    f(x) = log(exp(x) / sum(exp(x))) = log(softmax(x))

    Args:
        x (Tensor): tensor

    Returns:
        Tensor: result
    """
    return LogSoftmax.apply(x, dim)

# ******************************
# ******* Loss functions *******
# ******************************

class MSELoss(Function):
    
    @staticmethod
    def forward(ctx:Context, y_pred:Tensor, y_true:Tensor):
        if not isinstance(y_pred, Tensor):
            raise TypeError(f"Expected y_pred to be a Tensor but got {type(y_pred)}")
        if not isinstance(y_true, Tensor):
            raise TypeError(f"Expected y_true to be a Tensor but got {type(y_true)}")
        
        if not y_pred.matches_shape(y_true):
            raise ValueError(f"Inputs shape don't match y_pred={y_pred.shape}, y_true={y_true.shape}")
        
        if y_pred.device == Device.CPU:
            loss_data = cpu_ops.mse_loss_forward(y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {y_pred.device} not supported")

        loss = Tensor(loss_data, device=y_pred.device)

        ctx.save_for_backward(y_pred, y_true)
        
        return loss
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        y_pred, y_true = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            loss_grad_data = cpu_ops.mse_loss_backward(grad_output.data, y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        loss_grad = Tensor(loss_grad_data, device=grad_output.device)
        
        return loss_grad
    

def mse_loss(y_pred:Tensor, y_true:Tensor):
    """ 
    Mean Squared Error loss function.

    Args:
        y_pred (Tensor): tensor
        y_true (Tensor): tensor

    Returns:
        Tensor: result
    """
    return MSELoss.apply(y_pred, y_true)


class NLLLoss(Function):
    
    @staticmethod
    def forward(ctx:Context, y_pred:Tensor, y_true:Tensor):
        if not isinstance(y_pred, Tensor):
            raise TypeError(f"Expected y_pred to be a Tensor but got {type(y_pred)}")
        if not isinstance(y_true, Tensor):
            raise TypeError(f"Expected y_true to be a Tensor but got {type(y_true)}")
        
        if y_pred.device == Device.CPU:
            loss_data = cpu_ops.nll_loss_forward(y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {y_pred.device} not supported")

        loss = Tensor(loss_data, device=y_pred.device)

        ctx.save_for_backward(y_pred, y_true)
        
        return loss
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        y_pred, y_true = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            loss_grad_data = cpu_ops.nll_loss_backward(grad_output.data, y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        loss_grad = Tensor(loss_grad_data, device=grad_output.device)
        
        return loss_grad
    

def nll_loss(y_pred:Tensor, y_true:Tensor):
    """ 
    Negative Log Likelihood loss function.

    Args:
        y_pred (Tensor): tensor
        y_true (Tensor): tensor

    Returns:
        Tensor: result
    """
    return NLLLoss.apply(y_pred, y_true)


class BCELoss(Function):
    
    @staticmethod
    def forward(ctx:Context, y_pred:Tensor, y_true:Tensor):
        if not isinstance(y_pred, Tensor):
            raise TypeError(f"Expected y_pred to be a Tensor but got {type(y_pred)}")
        if not isinstance(y_true, Tensor):
            raise TypeError(f"Expected y_true to be a Tensor but got {type(y_true)}")
        
        if y_pred.device == Device.CPU:
            loss_data = cpu_ops.bce_loss_forward(y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {y_pred.device} not supported")

        loss = Tensor(loss_data, device=y_pred.device)

        ctx.save_for_backward(y_pred, y_true)
        
        return loss
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        y_pred, y_true = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            loss_grad_data = cpu_ops.bce_loss_backward(grad_output.data, y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        loss_grad = Tensor(loss_grad_data, device=grad_output.device)
        
        return loss_grad
    

def binary_cross_entropy(y_pred:Tensor, y_true:Tensor):
    """ 
    Binary Cross Entropy loss function.

    Args:
        y_pred (Tensor): tensor
        y_true (Tensor): tensor

    Returns:
        Tensor: result
    """
    return BCELoss.apply(y_pred, y_true)


class BCEWithLogitsLoss(Function):
    
    @staticmethod
    def forward(ctx:Context, y_pred:Tensor, y_true:Tensor):
        if not isinstance(y_pred, Tensor):
            raise TypeError(f"Expected y_pred to be a Tensor but got {type(y_pred)}")
        if not isinstance(y_true, Tensor):
            raise TypeError(f"Expected y_true to be a Tensor but got {type(y_true)}")
        
        if y_pred.device == Device.CPU:
            loss_data = cpu_ops.bce_with_logits_loss_forward(y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {y_pred.device} not supported")

        loss = Tensor(loss_data, device=y_pred.device)

        ctx.save_for_backward(y_pred, y_true)
        
        return loss
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        y_pred, y_true = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            loss_grad_data = cpu_ops.bce_with_logits_loss_backward(grad_output.data, y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        loss_grad = Tensor(loss_grad_data, device=grad_output.device)
        
        return loss_grad
    

def binary_cross_entropy_with_logits(y_pred:Tensor, y_true:Tensor):
    """ 
    Binary Cross Entropy with Logits loss function.

    Args:
        y_pred (Tensor): tensor
        y_true (Tensor): tensor

    Returns:
        Tensor: result
    """
    return BCEWithLogitsLoss.apply(y_pred, y_true)


class CrossEntropyLoss(Function):
    
    @staticmethod
    def forward(ctx:Context, y_pred:Tensor, y_true:Tensor):
        if not isinstance(y_pred, Tensor):
            raise TypeError(f"Expected y_pred to be a Tensor but got {type(y_pred)}")
        if not isinstance(y_true, Tensor):
            raise TypeError(f"Expected y_true to be a Tensor but got {type(y_true)}")
        
        if y_pred.device == Device.CPU:
            loss_data = cpu_ops.cross_entropy_loss_forward(y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {y_pred.device} not supported")

        loss = Tensor(loss_data, device=y_pred.device)

        ctx.save_for_backward(y_pred, y_true)
        
        return loss
    
    @staticmethod
    def backward(ctx:Context, grad_output:Tensor):
        y_pred, y_true = ctx.saved_tensors
        
        if grad_output.device == Device.CPU:
            loss_grad_data = cpu_ops.cross_entropy_loss_backward(grad_output.data, y_pred.data, y_true.data)
        else:
            raise RuntimeError(f"{ctx.fn_name}: {grad_output.device} not supported")
        
        loss_grad = Tensor(loss_grad_data, device=grad_output.device)
        
        return loss_grad
    

def cross_entropy(y_pred:Tensor, y_true:Tensor):
    """ 
    Cross Entropy loss function.

    Args:
        y_pred (Tensor): tensor
        y_true (Tensor): tensor

    Returns:
        Tensor: result
    """
    return CrossEntropyLoss.apply(y_pred, y_true)

# *******************************
# ******* Pool functions ********
# *******************************




# *******************************
# ******* Conv functions ********
# *******************************




# *************************************
# ******* Batch norm functions ********
# *************************************