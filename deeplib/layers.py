
from deeplib.engine import Tensor
from deeplib.neurons import Neuron
from deeplib.modules import Module


class Linear(Module):
    
    def __init__(self, input_size:int, output_size:int, weight_init_method='he'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = []
        for _ in range(self.output_size):
            n = Neuron(input_size, weight_init_method=weight_init_method)
            self.neurons.append(n)
        
    def forward(self, x:Tensor) -> Tensor:
        assert x.shape[1] == self.input_size, f"Expected input size '{self.input_size}' but received '{x.shape[1]}'"
        
        out = Tensor.concat([ neuron(x) for neuron in self.neurons ], dim=1)
        
        return out
    
    def parameters(self) -> list[Tensor]:
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self) -> str:
        return f"Linear(input_size={self.input_size}, neurons={len(self.neurons)})"
        
    
if __name__ == "__main__":
    list_ = [[0.2, 1, 4, 2, -1, 4], [0.2, 1, 4, 2, -1, 4]]
    inp = Tensor(list_)
    
    linear = Linear(6,2)
    linear.retain_grad()
    
    out = linear(inp)
    print(out)
    out = out.sum()
    out.backward()
    
    neuron = linear.neurons[0]
    print(neuron)
    print("Weights grad:\n", neuron.weights.grad)
    print("Bias grad:\n", neuron.bias.grad)
    
    import torch
    
    inp = torch.tensor(list_)
    linear = torch.nn.Linear(6,1)
    
    out = linear(inp)
    print(out)
    
# class Conv2D(Layer):
    
#     def __init__(self, filters, kernel_size, strides=None, padding=None) -> None:
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.padding = padding
        
#     def __call__(self, x:np.ndarray) -> np.ndarray:
#         super().__call__(x)
#         ...
    
# class BatchNorm2d(Layer):
#     ...
    
# class MaxPool2D(Layer):
    
#     def __call__(self, x:np.ndarray) -> np.ndarray:
#         super().__call__(x)
#         ...