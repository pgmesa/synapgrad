import math
import numpy as np
from synapgrad.tensor import Tensor


# Referece to pytorch code: 
# https://pytorch.org/docs/stable/nn.init.html

def calculate_gain(nonlinearity:str, param=None) -> float:
    """Return the recommended gain value for the given nonlinearity function.
    The values are as follows:
    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    linear            :math:`1`
    conv{1,2}d        :math:`1`
    sigmoid           :math:`1`
    tanh              :math:`5/3`
    relu              :math:`sqrt(2)`
    leaky_relu        :math:`sqrt(2 / (1 + negative_slope^2))`
    selu              :math:`3/4`
    ================= ====================================================
    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function
    """
    linear_fns = ['linear', 'conv1d', 'conv2d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def _calculate_fan_in_and_fan_out(tensor:Tensor) -> tuple[int, int]:
    """
    Calculate the fan_in and fan_out of a weight tensor.
    
    Args:
        tensor: an n-dimensional `synapgrad.Tensor`
    Returns:
        a tuple of (fan_in, fan_out)
    """
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        receptive_field_size = np.prod(tensor.shape[2:])

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def uniform_(tensor:Tensor, a=0.0, b=1.0) -> Tensor:
    """
    Fills the input Tensor with values drawn from the uniform
    distribution :math:`U(a,b)`.
    
    Args:
        tensor: an n-dimensional `synapgrad.Tensor`
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
    Examples:
        >>> w = synapgrad.empty(3, 5)
        >>> nn.init.uniform_(w)
    """
    tensor.data = np.random.uniform(a, b, tensor.shape).astype(tensor.dtype)
    return tensor
    return np.random.uniform(low=-scale, high=scale, size=shape)


def normal_(tensor:Tensor, mean=0.0, std=1.0) -> Tensor:
    """
    Fills the input Tensor with values drawn from the normal
    distribution :math:`N(mean,std^2)`.
    
    Args:
        tensor: an n-dimensional `synapgrad.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
    Examples:
        >>> w = synapgrad.empty(3, 5)
        >>> nn.init.normal_(w)
    """
    tensor.data = np.random.normal(mean, std, tensor.shape).astype(tensor.dtype)
    return tensor


def constant_(tensor:Tensor, val) -> Tensor:
    """
    Fills the input Tensor with the value `val`.
    
    Args:
        tensor: an n-dimensional `synapgrad.Tensor`
        val: the value to fill the tensor with
    Examples:
        >>> w = synapgrad.empty(3, 5)
        >>> nn.init.constant_(w)
    """
    tensor.data = np.full(tensor.shape, val).astype(tensor.dtype)
    return tensor


def ones_(tensor:Tensor) -> Tensor:
    """
    Fills the input Tensor with the value `1`.
    
    Args:
        tensor: an n-dimensional `synapgrad.Tensor`
    Examples:
        >>> w = synapgrad.empty(3, 5)
        >>> nn.init.ones_(w)
    """
    tensor.data = np.ones(tensor.shape).astype(tensor.dtype)
    return tensor


def zeros_(tensor:Tensor) -> Tensor:
    """
    Fills the input Tensor with the value `0`.
    
    Args:
        tensor: an n-dimensional `synapgrad.Tensor`
    Examples:
        >>> w = synapgrad.empty(3, 5)
        >>> nn.init.zeros_(w)
    """
    tensor.data = np.zeros(tensor.shape).astype(tensor.dtype)
    return tensor


def xavier_uniform_(tensor:Tensor, gain:float=1.0) -> Tensor:
    """ 
    Fills the input Tensor with values according to the method described in
    `Understanding the difficulty of training deep feedforward neural networks`
    - Xavier Glorot and Yoshua Bengio (2010), using a uniform distribution
    
    The resulting tensor will have values sampled from :math:`U(-a,a)` where
    :math:`a = gain * sqrt(6 / (fan_in + fan_out))`.
    
    Args:
        tensor: an n-dimensional `synapgrad.Tensor`
        gain: scaling factor
    Examples:
        >>> w = synapgrad.empty(3, 5)
        >>> nn.init.xavier_uniform_(w)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    a = gain * math.sqrt(6.0 / float(fan_in + fan_out))
    return uniform_(tensor, -a, a)


def xavier_normal_(tensor:Tensor, gain:float=1.0) -> Tensor:
    """
    Fills the input Tensor with values according to the method described in
    `Understanding the difficulty of training deep feedforward neural networks`
    - Xavier Glorot and Yoshua Bengio (2010), using a normal distribution
    
    The resulting tensor will have values sampled from :math:`N(0,std^2)` where
    :math:`std = gain * sqrt(2 / (fan_in + fan_out))`.
    
    Args:
        tensor: an n-dimensional `synapgrad.Tensor`
        gain: scaling factor
    Examples:
        >>> w = synapgrad.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return normal_(tensor, 0, std**2)


def kaiming_uniform_(tensor:Tensor, a=0, mode='fan_in', nonlinearity='leaky_relu') -> Tensor:
    """
    Fills the input Tensor with values according to the method described in
    `Delving deep into rectifiers: Surpassing human-level performance on
    ImageNet classification` - He et al. (2015), using a uniform distribution
    
    The resulting tensor will have values sampled from :math:`U(-a,a)` where
    :math:`a = gain * sqrt(3 / fan_mode)`
    
    Args:
        tensor: an n-dimensional `synapgrad.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either `fan_in` (default) or `fan_out`. Choosing `fan_in`
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing `fan_out` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (nn.functional name), 
            recommended to use only with 'relu' or 'leaky_relu' (default)
    Examples:
        >>> w = synapgrad.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w)
    """
    fan = _calculate_fan_in_and_fan_out(tensor)
    fans_str = ['fan_in', 'fan_out']
    if mode in fans_str:
        mode = fans_str.index(mode)
    else:
        raise ValueError(f"invalid {mode=} for kaiming uniform")
    
    gain = calculate_gain(nonlinearity, a)
    std = gain * math.sqrt(3.0 / float(fan[mode]))
    return uniform_(tensor, -std, std)


def kaiming_normal_(tensor:Tensor, a=0, mode='fan_in', nonlinearity='leaky_relu') -> Tensor:
    """
    Fills the input Tensor with values according to the method described in
    `Delving deep into rectifiers: Surpassing human-level performance on
    ImageNet classification` - He et al. (2015), using a normal distribution.
    
    The resulting tensor will have values sampled from :math:`N(0,std^2)` where
    :math:`std = gain * sqrt(2 / fan_mode)`
    
    Args:
        tensor: an n-dimensional `synapgrad.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either `fan_in` (default) or `fan_out`. Choosing `fan_in`
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing `fan_out` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (nn.functional name),
            recommended to use only with 'relu' or 'leaky_relu' (default).
    """
    fan = _calculate_fan_in_and_fan_out(tensor)
    fans_str = ['fan_in', 'fan_out']
    if mode in fans_str:
        mode = fans_str.index(mode)
    else:
        raise ValueError(f"invalid {mode=} for kaiming normal")
    
    gain = calculate_gain(nonlinearity, a)
    std = gain * (1 / math.sqrt(float(fan[mode])))
    return normal_(tensor, 0, std**2)