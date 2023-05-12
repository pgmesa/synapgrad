from synapgrad.autograd import no_grad, retain_grads, retain_children, retain_all
from synapgrad.tensor import (
    Tensor, tensor, ones, ones_like, zeros, zeros_like, 
    arange, rand, randn, normal, randint, eye
)
from synapgrad.functional import *
from synapgrad.tools import manual_seed

# from synapgrad.nn.functional import *

from synapgrad import autograd, cpu_ops, device, functional, visual, tensor, tools