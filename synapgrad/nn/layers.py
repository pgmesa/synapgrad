import numpy as np
import synapgrad
from synapgrad import nn
from synapgrad.tensor import Tensor
from synapgrad.nn import functional as F
from .initializations import init_weights


class Linear(nn.Module):
    
    def __init__(self, input_size:int, output_size:int, weight_init_method='he_normal'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.weight_init_method = weight_init_method
        weight_values = init_weights((output_size, input_size), weight_init_method).astype(np.float32)
        self.weight = synapgrad.tensor(weight_values, requires_grad=True)
        self.bias = synapgrad.zeros((output_size,), dtype=np.float32, requires_grad=True)
        
    def forward(self, x:Tensor) -> Tensor:
        assert x.shape[1] == self.input_size, f"Expected input size '{self.input_size}' but received '{x.shape[1]}'"

        out = (x @ self.weight.transpose(0,1)) + self.bias

        return out
    
    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]
    
    def __repr__(self) -> str:
        return f"Linear(input_size={self.input_size}, neurons={len(self.output_size)})"


class Flatten(nn.Module):
    
    def __init__(self, start_dim=1, end_dim=-1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)
    
    
class Unfold(nn.Module):
    """ 
    Check nn.funcional.unfold for more information 
    """
    
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, pad_value=0) -> None:
        super().__init__()
        
        kernel_size = np.broadcast_to(kernel_size, 2)
        dilation = np.broadcast_to(dilation, 2)
        padding = np.broadcast_to(padding, 2)
        stride = np.broadcast_to(stride, 2)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pad_value = pad_value
        
    def forward(self, x: Tensor) -> Tensor:
        return F.unfold(x.data, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                       dilation=self.dilation, pad_value=self.pad_value)
    
    
class Fold(nn.Module):
    """ 
    Check nn.funcional.fold for more information 
    """
    
    def __init__(self, output_size, kernel_size, stride=1, padding=0, dilation=1) -> None:
        super().__init__()
        
        output_size = np.broadcast_to(output_size, 2)
        kernel_size = np.broadcast_to(kernel_size, 2)
        dilation = np.broadcast_to(dilation, 2)
        padding = np.broadcast_to(padding, 2)
        stride = np.broadcast_to(stride, 2)
        
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
    def forward(self, x: Tensor) -> Tensor:
        return F.fold(x.data, output_size=self.output_size, kernel_size=self.kernel_size, stride=self.stride,
                      padding=self.padding, dilation=self.dilation)
    

class MaxPool2d(nn.Module):
    
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1) -> None:
        """
        Applies Max Pooling 2D to a batch of N images with C channels (N, C, H, W).
        References:
            https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

        Args:
            kernel_size (int or tuple): the size of the sliding blocks
            stride (int or tuple, optional): the stride of the sliding blocks in the input spatial dimensions. Default: 1
            padding (int or tuple, optional):  implicit zero padding to be added on both sides of input. Default: 0
            dilation (int or tuple, optional): a parameter that controls the stride of elements within the neighborhood. Default: 1

        Output:
            Tensor of shape (N, C, H // stride[0], W // stride[1]).
        """
        super().__init__()
        
        kernel_size = np.broadcast_to(kernel_size, 2)
        if stride is None:
            stride = kernel_size
        else:
            stride = np.broadcast_to(stride, 2)
        padding = np.broadcast_to(padding, 2)
        dilation = np.broadcast_to(dilation, 2)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
    def forward(self, x: Tensor) -> Tensor: 
        N, C, H, W = x.shape
        lH = int(np.floor((H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        lW = int(np.floor((W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        
        unfolded = unfold(x.data, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
                           padding=self.padding, pad_value=-np.inf)
        
        N, C_k, L = unfolded.shape
        split_per_channel = unfolded.reshape(N, C, int(C_k/C), L)
        
        unfolded_reorganized = np.moveaxis(split_per_channel, -2, -1).reshape(N, C, lH, lW, -1)
        pooled, max_indices, selected = Tensor(unfolded_reorganized).max(dim=-1, return_selected=True)
    
        out = Tensor(pooled.data, (x,), "<MaxPool2d>", requires_grad=x.requires_grad)
        
        def _backward():
            if x.requires_grad:
                grad = selected * np.expand_dims(out._grad, axis=-1)

                grad = grad.reshape(N, C, L, int(C_k/C))
                grad = np.moveaxis(grad, -1, -2)
                grad = grad.reshape(N, C_k, L)
                
                grad = fold(grad, (H,W), kernel_size=self.kernel_size, stride=self.stride,
                     dilation=self.dilation, padding=self.padding)

                x._grad += grad
                
        out._backward = _backward
        
        return out
    
    
class AvgPool2d(nn.Module):
    ...


class Conv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, weight_init_method='he_uniform') -> None:
        """
        Applies 2D Convolution to a batch of N images with C channels (N, C, H, W).
        Reference:
            https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        Args:
            in_channels (int): the number of input channels
            out_channels (int): the number of output channels produced by the convolution
            kernel_size (int or tuple): the size of the sliding blocks
            stride (int or tuple, optional): the stride of the sliding blocks in the input spatial dimensions. Default: 1
            padding (int or tuple, optional):  implicit zero padding to be added on both sides of input. Default: 0
            dilation (int or tuple, optional): a parameter that controls the stride of elements within the neighborhood. Default: 1
        """
        super().__init__()
        
        kernel_size = np.broadcast_to(kernel_size, 2)
        if stride is None:
            stride = kernel_size
        else:
            stride = np.broadcast_to(stride, 2)
        
        if padding == 'same':
            if any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")
            padding = int(np.floor(kernel_size[0] / 2))
        if padding == 'valid':
            padding = 0
        padding = np.broadcast_to(padding, 2)
        dilation = np.broadcast_to(dilation, 2)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.weight_init_method = weight_init_method
        weight_values = init_weights((out_channels, in_channels, *kernel_size), weight_init_method).astype(np.float32)
        self.weight = Tensor(weight_values, requires_grad=True)
        self.bias = Tensor.zeros((out_channels,), dtype=np.float32, requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        lH = int(np.floor((H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        lW = int(np.floor((W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        
        unfolded = Unfold(kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
                           padding=self.padding, pad_value=0)(x)
        
        weight = self.weight.view(self.weight.shape[0], -1).transpose(0,1) 
        mult = unfolded.transpose(1,2) @ weight
        convolved = (mult) + self.bias
        out = convolved.transpose(1,2).view(N, self.out_channels, lH, lW)
                
        return out
        
    def parameters(self) -> list['Tensor']:
        return [self.weight, self.bias]
    
    def __repr__(self) -> str:
        return (f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, " + 
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, " + 
                f"dilation={self.dilation}")
    

class Dropout(nn.Module):
    """ 
    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
    
    """

    def __init__(self, p=0.5, inplace=False) -> None:
        super().__init__()
        self.p = p
        self.inplace = inplace
        
    def forward(self, x: Tensor) -> Tensor:
        if not self.training: return x
        random_data = np.random.rand(*x.shape)
        random_data = np.where(random_data <= self.p, 0, 1)
        if self.p < 1:
            random_data = random_data / (1-self.p) # scale data
        random_t = synapgrad.tensor(random_data)
        
        return x*random_t
    
    
class BatchNorm(nn.Module):
    
    def __init__(self, mode:str, num_features:int, eps:float=1e-5, momentum:float=0.1, affine:bool=True,
                 track_running_stats:bool=True, dtype=None) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.num_batches_tracked = 0
        
        valid_modes = ['1d', '2d']
        
        if mode not in valid_modes:
            raise ValueError(f"'{mode}' is not a valid mode, expected one of {valid_modes}")
        self.mode = mode
        
        if dtype is None:
            dtype = np.float32
        
        if self.track_running_stats:
            self.running_mean = Tensor.zeros(num_features, dtype=dtype, name='running_mean')
            self.running_var = Tensor.ones(num_features, dtype=dtype, name='running_var')
        else:
            self.running_mean = None
            self.running_var = None
        
        if affine:
            # gamma
            self.weight = Tensor.ones(num_features, requires_grad=True, dtype=dtype, name="gamma")
            # beta
            self.bias = Tensor.zeros(num_features, requires_grad=True, dtype=dtype, name="beta")
                
    def forward(self, x: Tensor) -> Tensor:
        if self.mode == '2d':
            if len(x.shape) == 4:
                N, C, H, W = x.shape
                n = N*H*W # num_samples
                dims = (0,2,3)
                view_shape = (1,C,1,1)
            else: raise RuntimeError(f"Expected 4D tensor, but got {len(x.shape)}D")
        elif self.mode == '1d':
            if len(x.shape) == 3:
                N, C, L = x.shape
                n = N*L # num_samples
                dims = (0,2)
                view_shape = (1,C,1)
            elif len(x.shape) == 2:
                N, C = x.shape
                n = N # num_samples
                dims = (0,)
                view_shape = (1,C)
            else: raise RuntimeError(f"Expected 2D or 3D tensor, but got {len(x.shape)}D")
        assert C == self.num_features, f"Expected {self.num_features} channels, got {C}."
        
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = x.mean(dim=dims)
            # use biased var in train
            var_sum = ((x - mean.reshape(view_shape))**2).sum(dim=dims)
            var = var_sum / n
            
            r_mu = (exponential_average_factor * mean.data + (1 - exponential_average_factor) * self.running_mean.data)
            self.running_mean = Tensor(r_mu, dtype=x.dtype)
            
            unbiased_var = var_sum.data / (n - 1)
            r_var = (exponential_average_factor * unbiased_var + (1 - exponential_average_factor) * self.running_var.data)
            self.running_var = Tensor(r_var, dtype=x.dtype)
        else:
            mean = self.running_mean
            var = self.running_var

        out = (x - mean.reshape(view_shape)) / (var.reshape(view_shape) + self.eps).sqrt()
        if self.affine:
            out = out * self.weight.reshape(view_shape) + self.bias.reshape(view_shape)

        return out
        
    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias] if self.affine else []


class BatchNorm1d(BatchNorm):
    """
    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
    """
    
    def __init__(self, num_features:int, eps:float=1e-5, momentum:float=0.1, affine:bool=True,
                 track_running_stats:bool=True, dtype=None) -> None:
        """
        Computes the batchnorm of a 2 or 3 dimensional tensor (N, C) or (N, C, L) 

        Arguments:
        num_features (int): C from an expected input size (N, C) or (N, C, L).
        eps (float): small constant to avoid division by zero in variance computation.
        momentum (float): the value used for the running_mean and running_var computation.
            Can be set to None for cumulative moving average (i.e. simple average).
            Default: 0.1.
        affine (bool): a boolean value that when set to True, this module has 
            learnable affine parameters. Default: True
        track_running_stats (bool): a boolean value that when set to True, this module 
            tracks the running mean and variance, and when set to False, this 
            module does not track such statistics, and initializes statistics buffers
            running_mean and running_var as None. When these buffers are None, this 
            module always uses batch statistics. in both training and eval modes. 
            Default: True
        """
        super().__init__('1d', num_features, eps, momentum, affine, track_running_stats, dtype)

    
class BatchNorm2d(BatchNorm):
    """
    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    """
    
    def __init__(self, num_features:int, eps:float=1e-5, momentum:float=0.1, affine:bool=True,
                 track_running_stats:bool=True, dtype=None) -> None:
        """
        Computes the batchnorm of a 4-dimensional tensor (N, C, H, W) 

        Arguments:
        num_features (int): C from an expected input size (N, C, H, W).
        eps (float): small constant to avoid division by zero in variance computation.
        momentum (float): the value used for the running_mean and running_var computation.
            Can be set to None for cumulative moving average (i.e. simple average).
            Default: 0.1.
        affine (bool): a boolean value that when set to True, this module has 
            learnable affine parameters. Default: True
        track_running_stats (bool): a boolean value that when set to True, this module 
            tracks the running mean and variance, and when set to False, this 
            module does not track such statistics, and initializes statistics buffers
            running_mean and running_var as None. When these buffers are None, this 
            module always uses batch statistics. in both training and eval modes. 
            Default: True
        """
        super().__init__('2d', num_features, eps, momentum, affine, track_running_stats, dtype)
        
        

        
    
