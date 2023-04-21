import numpy as np
from .. import nn, Tensor
from .neurons import init_weights
from .functional import unfold, fold
from .initializations import init_weights


class Linear(nn.Module):
    
    def __init__(self, input_size:int, output_size:int, weight_init_method='he_normal'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.weight_init_method = weight_init_method
        weight_values = init_weights((output_size, input_size), weight_init_method).astype(np.float32)
        self.weights = Tensor(weight_values, requires_grad=True)
        self.bias = Tensor(np.zeros((output_size,), dtype=np.float32), requires_grad=True)
        
    def forward(self, x:Tensor) -> Tensor:
        assert x.shape[1] == self.input_size, f"Expected input size '{self.input_size}' but received '{x.shape[1]}'"

        out = (x @ self.weights.transpose(0,-1)) + self.bias
        
        return out
    
    def parameters(self) -> list[Tensor]:
        return [self.weights, self.bias]
    
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
    """ Check nn.funcional.unfold for more information """
    
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
        N, C, H, W = x.shape
        unfolded = unfold(x.data, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
                           padding=self.padding, pad_value=self.pad_value)
        
        out = Tensor(unfolded, (x,), "<Unfold>", requires_grad=x.requires_grad)
        
        def _backward():
            if x.requires_grad:
                grad = fold(out._grad, (H,W), kernel_size=self.kernel_size, stride=self.stride,
                     dilation=self.dilation, padding=self.padding)

                x._grad += grad
                
        out._backward = _backward
    
        return out
    
    
class Fold(nn.Module):
    """ Check nn.funcional.fold for more information """
    
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
        folded = fold(x.data, self.output_size, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
                           padding=self.padding)
        
        out = Tensor(folded, (x,), "<Fold>", requires_grad=x.requires_grad)
        
        def _backward():
            if x.requires_grad:
                grad = unfold(out._grad, kernel_size=self.kernel_size, stride=self.stride,
                     dilation=self.dilation, padding=self.padding)

                x._grad += grad
                
        out._backward = _backward
    
        return out
    

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
        self.weights = Tensor(weight_values, requires_grad=True)
        self.bias = Tensor(np.zeros((out_channels,), dtype=np.float32), requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        lH = int(np.floor((H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        lW = int(np.floor((W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        
        unfolded = Unfold(kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
                           padding=self.padding, pad_value=0)(x)
        
        weights = self.weights.view(self.weights.shape[0], -1).transpose(0,1) 
        print("Matmul shapes", unfolded.transpose(1,2).shape, weights.shape)
        mult = unfolded.transpose(1,2) @ weights
        print("Result", mult.shape)
        convolved = (mult) + self.bias
        out = convolved.transpose(1,2).view(N, self.out_channels, lH, lW)

        return out
        
    def parameters(self) -> list['Tensor']:
        return [self.weights, self.bias]
    
    def __repr__(self) -> str:
        return (f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, " + 
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, " + 
                f"dilation={self.dilation}")
    
    
class BatchNorm2d(nn.Module):
    """
    TODO: Gradient not working properly
    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    """
    
    def __init__(self, num_features:int, eps:float=1e-5, momentum:float=0.1, affine:bool=True,
                 track_running_stats:bool=True) -> None:
        """
        Computes the batchnorm2d of a 4-dimensional tensor and its gradient.

        Arguments:
        x: tensor of shape (N, C, H, W) representing a batch of images.
        gamma: tensor of shape (C,) representing the normalization scale.
        beta: tensor of shape (C,) representing the normalization bias.
        eps: small constant to avoid division by zero in variance computation.
        momentum: exponential moving average factor for the mean and variance.
        track_running_stats: 
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        self.running_mean = None
        self.running_var = None
        
        if affine:
            self.gamma = Tensor(np.ones(num_features), requires_grad=True, dtype=np.float32)
            self.beta = Tensor(np.zeros(num_features), requires_grad=True, dtype=np.float32)
            
            
    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        num_examples = N*H*W
        assert C == self.num_features, f"Expected {self.num_features} channels, got {C}."
        # Compute the mean of each channel
        mu = x.mean(dim=(0,2,3), keepdims=True)

        # Compute the variance of each channel
        var = ((x - mu)**2).mean(dim=(0,2,3), keepdims=True)
        std = (var + self.eps).sqrt()

        # Update the running average of mean and variance
        if self.track_running_stats:
            if self.running_mean is None:
                self.running_mean = mu.detach().squeeze()
            else:
                self.running_mean = (self.momentum * self.running_mean + (1 - self.momentum) * mu.detach()).squeeze()
            
            if self.running_var is None:
                self.running_var = var.detach().squeeze()
            else:
                self.running_var = (self.momentum * self.running_var + (1 - self.momentum) * var.detach()).squeeze()

        # Normalize the input tensor
        x_norm = (x - mu) / std

        # Scale and shift the normalization
        if self.affine:
            out = self.gamma.view(1,C,1,1) * x_norm + self.beta.view(1,C,1,1)
        else:
            out = x_norm
        
        return out
        
    # def forward(self, x: Tensor) -> Tensor:
    #     N, C, H, W = x.shape
    #     num_examples = N*H*W
    #     assert C == self.num_features, f"Expected {self.num_features} channels, got {C}."
    #     # Compute the mean of each channel
    #     mu = np.mean(x.data, axis=(0,2,3), keepdims=True)

    #     # Compute the variance of each channel
    #     var = np.mean((x.data - mu)**2, axis=(0,2,3), keepdims=True)
    #     std = np.sqrt(var + self.eps)

    #     # Update the running average of mean and variance
    #     if self.track_running_stats:
    #         if self.running_mean is None:
    #             self.running_mean = mu
    #         else:
    #             self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            
    #         if self.running_var is None:
    #             self.running_var = var
    #         else:
    #             self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

    #     # Normalize the input tensor
    #     x_minus_mean = (x.data - mu) 
    #     x_norm = x_minus_mean / np.sqrt(var + self.eps)

    #     # Scale and shift the normalization
    #     if self.affine:
    #         out_data = self.gamma.data.reshape(1,C,1,1) * x_norm + self.beta.data.reshape(1,C,1,1)
    #     else:
    #         out_data = x_norm
        
    #     chidren = (x,) if not self.affine else (x, self.gamma, self.beta)
    #     out = Tensor(out_data, chidren, "<BatchNorm2d>", requires_grad=x.requires_grad)
        
    #     def _backward():
    #         grad = out._grad
            
    #         # Update gamma and beta gradient
    #         if self.affine:
    #             dgamma = np.sum(grad * x_norm, axis=(0,2,3))
    #             self.gamma._grad += dgamma
    #             dbeta = np.sum(grad, axis=(0,2,3))
    #             self.beta._grad += dbeta

    #         if x.requires_grad:  
    #             # standard_grad = grad

    #             # if self.affine: standard_grad *= self.gamma.data.reshape(1,C,1,1)

    #             # var_grad = np.sum(standard_grad * x_minus_mean * -0.5 * var ** (-3/2), axis=(0,2,3), keepdims=True)
    #             # stddev_inv = 1 / std
    #             # aux_x_minus_mean = 2 * x_minus_mean / num_examples

    #             # mean_grad = (np.sum(standard_grad * -stddev_inv, axis=(0,2,3), keepdims=True) + var_grad * np.sum(-aux_x_minus_mean, axis=(0,2,3), keepdims=True))

    #             # x._grad += standard_grad * stddev_inv + var_grad * aux_x_minus_mean + mean_grad / num_examples
                
    #             dx_norm = grad
    #             if self.affine: standard_grad *= self.gamma.data.reshape(1,C,1,1)
    #             dvar = np.sum(dx_norm * (x.data - mu) * -0.5 * (var + self.eps)**(-3/2), axis=(0,2,3), keepdims=True)
    #             dmu = np.sum(dx_norm * -1 / np.sqrt(var + self.eps), axis=(0,2,3), keepdims=True)
    #             dx = (dx_norm / np.sqrt(var + self.eps))    +    (dvar * 2 * (x.data - mu) / (N*H*W))   +   (dmu / (N*H*W))
                
    #             x._grad += dx
                 
    #             # if self.affine:
    #             #     dx_norm = grad * self.gamma.data.reshape(1,C,1,1) 
    #             #     dvar = np.sum(dx_norm * (x.data - mu) * -0.5 * (var + self.eps)**(-3/2), axis=(0,2,3), keepdims=True)
    #             #     dmu = np.sum(dx_norm * -1 / np.sqrt(var + self.eps), axis=(0,2,3), keepdims=True)
    #             #     dx = (dx_norm / np.sqrt(var + self.eps))    +    (dvar * 2 * (x.data - mu) / (N*H*W))   +   (dmu / (N*H*W))
    #             # else:
    #             #     dx = grad / np.sqrt(var + self.eps)
    #             #     print(np.max(dx))            
                
    #     out._backward = _backward
        
    #     return out

    def parameters(self) -> list[Tensor]:
        return [self.gamma, self.beta] if self.affine else []

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
        random_t = Tensor(random_data)
        
        return x*random_t