import numpy as np

# Referece to keras code: 
# https://github.com/keras-team/keras/blob/998efc04eefa0c14057c1fa87cab71df5b24bf7e/keras/initializations.py


weight_initializers = ['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_uniform']


def init_weights(shape, method) -> np.ndarray:
    if method not in weight_initializers:
        raise ValueError(f"'{method}' is not a valid weight initializer")
    
    if method == 'glorot_uniform': return glorot_uniform(shape)
    elif method == 'glorot_normal': return glorot_normal(shape)
    elif method == 'he_uniform': return he_uniform(shape)
    elif method == 'he_normal': return he_normal(shape)
    elif method == 'lecun_uniform': return lecun_uniform(shape)


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


def uniform(shape, scale=0.05) -> np.ndarray:
    return np.random.uniform(low=-scale, high=scale, size=shape)


def normal(shape, scale=0.05) -> np.ndarray:
    return np.random.normal(loc=0.0, scale=scale, size=shape)


def lecun_uniform(shape) -> np.ndarray:
    ''' Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    '''
    fan_in, fan_out = get_fans(shape)
    scale = np.sqrt(3. / fan_in)
    return uniform(shape, scale)


def glorot_normal(shape) -> np.ndarray:
    ''' Reference: Glorot & Bengio, AISTATS 2010
    '''
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, s)


def glorot_uniform(shape) -> np.ndarray:
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s)


def he_normal(shape) -> np.ndarray:
    ''' Reference:  He et al., http://arxiv.org/abs/1502.01852
    '''
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / fan_in)
    return normal(shape, s)


def he_uniform(shape) -> np.ndarray:
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / fan_in)
    return uniform(shape, s)


def orthogonal(shape, scale=1.1) -> np.ndarray:
    ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return scale * q[:shape[0], :shape[1]]


def identity(shape, scale=1) -> np.ndarray:
    if len(shape) != 2 or shape[0] != shape[1]:
        raise Exception('Identity matrix initialization can only be used '
                        'for 2D square matrices.')
    else:
        return scale * np.identity(shape[0])