from typing import Any
from collections import OrderedDict
from synapgrad.tensor import Tensor
from synapgrad.device import Device


class Parameter(Tensor):
    """
    Wrapper class for Tensor that allows it to be used as a module parameter.
    """ 
    def __repr__(self) -> str:
        return f"Parameter containing:\n{super().__repr__()}"


class Module:
    
    def __init__(self) -> None:
        self._submodules = OrderedDict()
        self._parameters = OrderedDict()
        self._initialized = True
        self.training = True
        
    def train(self):
        """ Set module and submodules to train mode"""
        self.training = True
        for m in self.submodules():
            m.train()
        return self
        
    def eval(self):
        """ Set module and submodules to evaluation mode """
        self.training = False
        for m in self.submodules():
            m.eval()
        return self
        
    def __call__(self, *inputs:Tensor, **kwargs) -> Tensor:
        return self.forward(*inputs, **kwargs)

    def zero_grad(self):
        for p in self.parameters():
            if p.requires_grad: p.zero_()
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
    
    def check_is_initialized(self):
        if not hasattr(self, '_initialized') or not self._initialized:
            raise RuntimeError("Module is not initialized. Call super().__init__() first")
        
    def register_module(self, name:str, module:'Module'):
        self.check_is_initialized()
    
        if not isinstance(module, Module):
            raise TypeError("All submodules must be of type Module")
        
        self._submodules[name] = module
        object.__setattr__(self, name, module)
        
    def register_parameter(self, name:str, parameter:Parameter):
        self.check_is_initialized()
        
        if not isinstance(parameter, Parameter):
            raise TypeError("All parameters must be of type Parameter")
        
        self._parameters[name] = parameter
        object.__setattr__(self, name, parameter)
        
    def apply(self, fn):
        """ 
        Applies fn recursively to every submodule as well as self. 
        Typical use includes initializing the parameters of a model
        """
        self.check_is_initialized()
        
        fn(self)
        for m in self.submodules():
            m.apply(fn)
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        """
        Keeps track of the submodules and parameters added to this module 
        """
        if isinstance(__value, Module):
            self.register_module(__name, __value)
        elif isinstance(__value, Parameter):
            self.register_parameter(__name, __value)
        else:  
            object.__setattr__(self, __name, __value)
    
    def parameters(self) -> list['Parameter']:
        """
        Returns a list of all parameters in the module
        """
        params = list(self._parameters.values())
        for m in self.submodules():
            params += m.parameters()
        return params
    
    def submodules(self) -> list['Module']:
        return [m for m in self._submodules.values()]
    
    def num_params(self, trainable=False, non_trainable=False) -> int:
        num_params = 0; num_trainable = 0; num_non_trainable = 0
        for p in self.parameters():
            num_params += p.size
            if p.requires_grad: num_trainable += p.size
            else: num_non_trainable += p.size
        
        if trainable: return num_trainable
        elif non_trainable: return num_non_trainable
        return num_params
    
    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("All subclasses of Module must implement forward method")
    
    def cpu(self):
        """
        Move all parameters and submodules to CPU
        """
        for p in self.parameters():
            if p.device != Device.CPU:
                p.cpu()
        for module in self.submodules():
            if p.device != Device.CPU:
                module.cpu()
        return self
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(submodules={len(self.submodules())}, " +
                f"parameters={self.num_params()}, trainable={self.num_params(trainable=True)}, " +
                f"non_trainable={self.num_params(non_trainable=True)})")
        
        
class Sequential(Module):
    
    def __init__(self, *modules) -> None:
        super().__init__()
        if len(modules) == 1 and isinstance(modules[0], OrderedDict):
            for key, module in modules[0].items():
                self.register_module(key, module)
        else:
            for idx, module in enumerate(modules):
                self.register_module(str(idx), module)
        
    def forward(self, x:Tensor) -> Tensor:
        inp = x
        for module in self.submodules():
            out = module(inp)
            inp = out
        return out
        
