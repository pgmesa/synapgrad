import math
import numpy as np
import synapgrad
from synapgrad import nn
from synapgrad.nn.modules import Parameter
from synapgrad.tensor import Tensor
from synapgrad.nn import functional as F
from synapgrad.nn import init


class Linear(nn.Module):
    
    def __init__(self, in_features:int, out_features:int, bias=True):
        """ 
        Applies a linear transformation to the incoming data: y = x @ w.T + b. 
        
        This layer is also known as Dense layer.
        
        Reference:
            - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        
        Args:
            in_features (int): The number of features in the input tensor.
            out_features (int): The number of features in the output tensor.
            weight_init_method (str): The method to use for initializing the weights.
                Options are 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_uniform'.
                Defaults to 'he_normal'.
            bias: Whether to use a bias or not. Defaults to True.
        
        Returns:
            A tensor of shape (batch_size, output_size)
        
        Variables:
            - weight (synapgrad.Tensor) - the learnable weights of the module of shape
                (out_features, in_features). The values are initialized from `U(-sqrt(k), sqrt(k))`
                where `k = 1/in_features`.
            - bias (synapgrad.Tensor) - the learnable bias of the module of shape (out_features).
                If `bias=True`, the values are initialized from `U(-sqrt(k), sqrt(k))` where
                `k = 1/in_features`.
        
        Notes:
            - The input tensor is expected to have a shape of (batch_size, input_size).
            - The output tensor is expected to have a shape of (batch_size, output_size).
        
        Example:
            >>> layer = nn.Linear(input_size=3, output_size=4)
            >>> x = synapgrad.ones((2, 3))
            >>> y = layer(x)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
      
        self.weight = nn.Parameter(synapgrad.empty((out_features, in_features), dtype=np.float32, requires_grad=True, name='weight'))
        if bias: 
            self.bias = nn.Parameter(synapgrad.empty((out_features,), dtype=np.float32, requires_grad=True, name='bias'))
        else:
            self.bias = None
        self.reset_parameters()
        
    def reset_parameters(self):
        """ 
        Resets the parameters of the layer.
        """
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        std = 1. / math.sqrt(float(fan_in)) if fan_in > 0 else 0
        init.uniform_(self.weight, -std, std)
        if self.bias is not None:
            init.uniform_(self.bias, -std, std)
        
    def forward(self, x:Tensor) -> Tensor:
        assert x.shape[1] == self.in_features, f"Expected input size '{self.in_features}' but received '{x.shape[1]}'"
        return F.linear(x, self.weight, self.bias)


class Neuron(Linear):

    def __init__(self, in_features: int, bias=True):
        """ 
        Creates a single Neuron layer. It's just a linear layer with out_features = 1
        
        Reference:
            - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        
        Args:
            input_size (int): The number of features in the input tensor.
            bias: Whether to use a bias or not. Defaults to True.
         
        Returns:
            A tensor of shape (batch_size, 1)
        
        Example:
            >>> layer = nn.Neuron(input_size=3)
            >>> x = synapgrad.ones((2, 3))
        """
        super().__init__(in_features, 1, bias=bias)
    

class Flatten(nn.Module):
    
    def __init__(self, start_dim=1, end_dim=-1) -> None:
        """ 
        Flattens a tensor over the specified start and end dimensions
        
        Reference:
            - https://pytorch.org/docs/stable/generated/torch.flatten.html
            
        Args:
            start_dim (int): The dimension to start flattening.
            end_dim (int): The dimension to end flattening.
        
        Returns:
            A tensor of shape (batch_size, *output_shape)
        
        Example:
            >>> layer = nn.Flatten()
            >>> x = synapgrad.ones((2, 3, 4, 5))
            >>> y = layer(x)
        """
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)
    

class Dropout(nn.Module):

    def __init__(self, p=0.5) -> None:
        """
        Randomly zeroes some of the elements of the input tensor with probability p
        using samples from a Bernoulli distribution. The values are also scaled by 1/(1-p)
        
        Reference:
            - https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
            
        Args:
            p (float): The probability of an element to be zeroed.
        
        Returns:
            A tensor of the same shape as the input tensor.
        
        Example:
            >>> layer = nn.Dropout(p=0.5)
            >>> x = synapgrad.ones((2, 3))
            >>> y = layer(x)
        """
        super().__init__()
        self.p = p
        
    def forward(self, x: Tensor) -> Tensor:
        if not self.training: return x
        random_data = np.random.rand(*x.shape)
        random_data = np.where(random_data <= self.p, 0, 1)
        if self.p < 1:
            random_data = random_data / (1-self.p) # scale data
        random_t = synapgrad.tensor(random_data)
        
        return x*random_t
    
    
class Unfold(nn.Module):
    
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, pad_value=0) -> None:
        """ 
        Unfolds tensor. See nn.functional.unfold for more information
        
        Reference:
            - https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
            
        Args:
            kernel_size (int or tuple): The size of the sliding window.
            stride (int or tuple): The stride of the sliding window.
            padding (int or tuple): The padding of the sliding window.
            dilation (int or tuple): The dilation of the sliding window.
            pad_value: The value to fill the padded area with.
        
        Example:
            >>> layer = nn.Unfold(kernel_size=3, stride=1, padding=1)
            >>> x = synapgrad.ones((2, 3, 4, 5))
            >>> y = layer(x) # Shape = (2, 27, 20)
        """
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
        return F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                       dilation=self.dilation, pad_value=self.pad_value)
    
    
class Fold(nn.Module):
    
    def __init__(self, output_size, kernel_size, stride=1, padding=0, dilation=1) -> None:
        """ 
        Folds a tensor. See nn.functional.fold for more information
        
        Args:
            kernel_size (int or tuple): The size of the sliding window.
            stride (int or tuple): The stride of the sliding window.
            padding (int or tuple): The padding of the sliding window.
            dilation (int or tuple): The dilation of the sliding window.
            output_size (int or tuple): The size of the output.
        
        Example:
            >>> layer = nn.Fold(output_size=(4,5), kernel_size=3, stride=1, padding=1)
            >>> x = synapgrad.ones((2, 27, 20))
            >>> y = layer(x) # Shape = (2, 3, 4, 5)
        
        """
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
        return F.fold(x, output_size=self.output_size, kernel_size=self.kernel_size, stride=self.stride,
                      padding=self.padding, dilation=self.dilation)


class MaxPool1d(nn.Module):
    
    def __init__(self, kernel_size:int, stride:int=None, padding:int=0, dilation:int=1) -> None:
        """
        Applies Max Pooling 1D to a batch of N images with C channels (N, C, W).
        
        Reference:
            - https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html

        Args:
            kernel_size (int): the size of the sliding blocks
            stride (int, optional): the stride of the sliding blocks in the input 
                spatial dimensions. If None, it is set to kernel_size. Default: None
            padding (int, optional): implicit zero padding to be added on both 
                sides of input. Default: 0
            dilation (int, optional): controls the spacing between the kernel points. Default: 1

        Output:
            Tensor of shape (N, C, lW).
        """
        super().__init__()
        
        self.kernel_size = kernel_size
        if stride is None: stride = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
    def forward(self, x: Tensor) -> Tensor: 
        return F.max_pool1d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                dilation=self.dilation)
        

class MaxPool2d(nn.Module):
    
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1) -> None:
        """
        Applies Max Pooling 2D to a batch of N images with C channels (N, C, H, W).
        References:
            https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html

        Args:
            kernel_size (int or tuple): the size of the sliding blocks
            stride (int or tuple, optional): the stride of the sliding blocks in the input 
                spatial dimensions. Default: 1
            padding (int or tuple, optional): implicit zero padding to be added on both 
                sides of input. Default: 0
            dilation (int or tuple, optional): controls the spacing between the kernel points. Default: 1

        Output:
            Tensor of shape (N, C, lW, lH).
        """
        super().__init__()
        
        kernel_size = np.broadcast_to(kernel_size, 2)
        if stride is None: stride = kernel_size
        else: stride = np.broadcast_to(stride, 2)
        padding = np.broadcast_to(padding, 2)
        dilation = np.broadcast_to(dilation, 2)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
    def forward(self, x: Tensor) -> Tensor: 
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                dilation=self.dilation)


class AvgPool1d(nn.Module):
    
    def __init__(self, kernel_size:int, stride:int=None, padding:int=0, dilation:int=1) -> None:
        """ 
        Applies Avg Pooling 1D to a batch of N images with C channels (N, C, W).
        
        Reference:
            - https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html

        Args:
            kernel_size (int): the size of the sliding blocks
            stride (int, optional): the stride of the sliding blocks in the input 
                spatial dimensions. If None, it is set to kernel_size. Default: None
            padding (int, optional): implicit zero padding to be added on both 
                sides of input. Default: 0
            dilation (int, optional): controls the spacing between the kernel points. Default: 1

        Output:
            Tensor of shape (N, C, lW).
        """
        super().__init__()
        self.kernel_size = kernel_size
        if stride is None: stride = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
    def forward(self, x: Tensor) -> Tensor: 
        return F.avg_pool1d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                dilation=self.dilation)

    
class AvgPool2d(nn.Module):
    
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1) -> None:
        """ 
        Applies Avg Pooling 2D to a batch of N images with C channels (N, C, H, W).
        
        Reference:
            - https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html

        Args:
            kernel_size (int or tuple): the size of the sliding blocks
            stride (int or tuple, optional): the stride of the sliding blocks in the input 
                spatial dimensions. If None, it is set to kernel_size. Default: None
            padding (int or tuple, optional): implicit zero padding to be added on both 
                sides of input. Default: 0
            dilation (int or tuple, optional): controls the spacing between the kernel points. Default: 1

        Output:
            Tensor of shape (N, C, lW, lH).
        """
        super().__init__()
        
        kernel_size = np.broadcast_to(kernel_size, 2)
        if stride is None: stride = kernel_size
        else: stride = np.broadcast_to(stride, 2)
        padding = np.broadcast_to(padding, 2)
        dilation = np.broadcast_to(dilation, 2)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
    def forward(self, x: Tensor) -> Tensor: 
        return F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                dilation=self.dilation)


class Conv1d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True) -> None:
        """
        Applies 1D Convolution to a batch of N images with C channels (N, C, W).
        
        Reference:
            - https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

        Args:
            in_channels (int): the number of input channels
            out_channels (int): the number of output channels produced by the convolution
            kernel_size (int): the size of the sliding blocks
            stride (int, optional): the stride of the sliding blocks in the input 
                spatial dimensions. Default: 1
            padding (int, optional): implicit zero padding to be added on both sides
                of input. Default: 0
            dilation (int, optional): controls the spacing between the kernel points. Default: 1
            bias: Whether to use a bias or not. Defaults to True.
        
        Output:
            Tensor of shape (N, C_out, lW).
        """
        super().__init__()
        
        if padding == 'same':
            if stride != 1:
                raise ValueError("padding='same' is not supported for strided convolutions")
            padding = int(np.floor(kernel_size / 2))
        if padding == 'valid':
            padding = 0
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        weight = synapgrad.empty((out_channels, in_channels, kernel_size), dtype=np.float32, requires_grad=True, name='weight')
        self.weight = nn.Parameter(weight)
        if bias: 
            bias = synapgrad.empty((out_channels,), dtype=np.float32, requires_grad=True, name='bias')
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1. / math.sqrt(float(fan_in)) if fan_in > 0 else 0
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: Tensor) -> Tensor:
        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


class Conv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True) -> None:
        """
        Applies 2D Convolution to a batch of N images with C channels (N, C, H, W).
        
        Reference:
            - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        Args:
            in_channels (int): the number of input channels
            out_channels (int): the number of output channels produced by the convolution
            kernel_size (int or tuple): the size of the sliding blocks
            stride (int or tuple, optional): the stride of the sliding blocks in the input 
                spatial dimensions. Default: 1
            padding (int or tuple, optional): implicit zero padding to be added on both sides
                of input. Default: 0
            dilation (int or tuple, optional): controls the spacing between the kernel points. Default: 1
            bias: Whether to use a bias or not. Defaults to True.
            
        Variables:
            - weight (Tensor): the learnable weights of the module of shape (C_out, C_in, kH, kW). The
                values of these weights are sampled from `U(-sqrt(k), sqrt(k))` where k = 1 / (C_in * kH * kW)
            - bias (Tensor): the learnable bias of the module of shape (C_out). If bias is True, then the values
                are initialized sampling from `U(-sqrt(k), sqrt(k))` where k = 1 / (C_in * kH * kW)
        
        Output:
            Tensor of shape (N, C_out, lW, lH).
        """
        super().__init__()
        
        kernel_size = np.broadcast_to(kernel_size, 2)
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
        
        weight = synapgrad.empty((out_channels, in_channels, kernel_size[0], kernel_size[1]), dtype=np.float32, requires_grad=True, name='weight')
        self.weight = nn.Parameter(weight)
        if bias:
            bias = synapgrad.empty((out_channels,), dtype=np.float32, requires_grad=True, name='bias') 
            self.bias = nn.Parameter(bias)
        else: 
            self.bias = None
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1. / math.sqrt(float(fan_in)) if fan_in > 0 else 0
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)
    
    
class BatchNorm(nn.Module):
    
    def __init__(self, num_features:int, eps:float=1e-5, momentum:float=0.1, affine:bool=True,
                 track_running_stats:bool=True, dtype=None) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.num_batches_tracked = 0

        if dtype is None:
            dtype = np.float32
        
        if self.track_running_stats:
            self.running_mean = synapgrad.zeros(num_features, dtype=dtype, name='running_mean')
            self.running_var = synapgrad.ones(num_features, dtype=dtype, name='running_var')
        else:
            self.running_mean = None
            self.running_var = None
        
        if affine:
            # gamma
            gamma = synapgrad.empty(num_features, requires_grad=True, dtype=dtype, name="gamma")
            self.weight = nn.Parameter(gamma)
            # beta
            beta = synapgrad.empty(num_features, requires_grad=True, dtype=dtype, name="beta")
            self.bias = nn.Parameter(beta)
            # Initialize parameters
            self.reset_parameters()
        else:
            self.weight = None
            self.bias = None
            
    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)
                
    def forward(self, x: Tensor) -> Tensor:
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
                    
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        
        running_mean = self.running_mean if not self.training or self.track_running_stats else None
        running_var = self.running_var if not self.training or self.track_running_stats else None
            
        return F.batch_norm(x, self.weight, self.bias, running_mean, running_var,
                                                bn_training, exponential_average_factor, self.eps)


class BatchNorm1d(BatchNorm):

    def __init__(self, num_features:int, eps:float=1e-5, momentum:float=0.1, affine:bool=True,
                 track_running_stats:bool=True, dtype=None) -> None:
        """
        Computes the batchnorm of a 2 or 3 dimensional tensor (N, C) or (N, C, L) 
        
        Reference:
            - https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

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
        super().__init__(num_features, eps, momentum, affine, track_running_stats, dtype)

    
class BatchNorm2d(BatchNorm):

    def __init__(self, num_features:int, eps:float=1e-5, momentum:float=0.1, affine:bool=True,
                 track_running_stats:bool=True, dtype=None) -> None:
        """
        Computes the batchnorm of a 4-dimensional tensor (N, C, H, W) 

        Reference:
            - https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

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
        super().__init__(num_features, eps, momentum, affine, track_running_stats, dtype)