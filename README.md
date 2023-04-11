#  SynapGrad

An autograd Tensor-based engine with a deep learning library built on top of it made from scratch

## Technical Description
This project implements a completely functional engine for tracking operations between Tensors, by dynamically building a Directed Acyclic Graph (DAG), and an automatic backpropagation algorithm (reverse-mode autodiff) over this DAG.

Built on top of the engine, the deep learning library implements the most common functions, layers, losses and optimizers in order to create MLPs and CNNs able to solve AI problems

This library tries to mimic Pytorch in a simplified way, but with similar functions and behaviour. 

## Aim of the project
The aim of this project is to create a deep learning library from scratch, without using any existing framework (such as keras, pytorch, tensorflow, sklearn, etc) in order to fully understand the core aspects of how they work. Specifically, this time I have focused on pytorch.

Some of the external frameworks mentioned before have been used for the following reasons:

- (pytorch) to check gradient calculation is correct
- (sklearn) (keras) to download the example datasets

This project stems from the amazing educational project `micrograd` by `karpathy` (https://github.com/karpathy/micrograd)

Note: Supporting GPU execution is out of the scope of this project

## Installation
```bash
pip install synapgrad
```

## Autograd Example
Below is a random example of some of the operations that can be tracked with synapgrad.Tensor
```python
from synapgrad import Tensor

l1 = [[-4.0, 0, 5.0], [6.3, 3.2, 1.3]]
l2 = [[2.0, 2,  3.0], [2.4, 1.7, 0.5]]

a = Tensor(l1, requires_grad=True).unsqueeze(0)**2
a.retain_grad()
b = 2**Tensor(l2, requires_grad=True).unsqueeze(0)
b.retain_grad()
c = Tensor(4.0, requires_grad=True)

out1 = Tensor.stack((a.squeeze(), b.squeeze()))[0]
out2 = Tensor.concat((a*c, b), dim=1).transpose(0, 1)[0, :]
out = out1 @ out2.view(3).unsqueeze(1)
print(out) # outcome of this forward pass
out.sum().backward()

print(a.grad) # da/dout
print(c.grad) # dc/dout
```

## Training examples using synapgrad

This project comes with 3 jupyter notebooks (in `examples/`) that solve 3 beginner's problems in AI:

- [x] 1. Basic MLP for binary classification (sklearn 'make_moons' toy dataset)
- [ ] 2. MLP for handwritten digits classification (MNIST dataset) 
- [ ] 3. CNN for handwritten digits classification (MNIST dataset)

So far, only the first example has been implemented

Example 1 (synapgrad MLP solution)     |  Example 2 and 3
:-------------------------:|:-------------------------:
![Board Image](/assets/example1.png) | ![Check Image](/assets/example23.png) 

## Comparisons with other frameworks
In order to see the efficiency of synapgrad, it is compared with other existing engines (in this case torch and micrograd).


| Training Example | synapgrad | torch | micrograd |
|     :---:        |  :---:  |  :---:  |   :---:   |  
| 1  | 1.7 s | 1.5 s | 1 min y 43 s |
| 2  | - | - | - |
| 3  | - | - | - |

As you can see, synapgrad is fast

## Graph Visualization
In the `examples/trace_graph.ipynb` notebook there is an example of how to display the graph that synapgrad creates in the background as operations are chained.

```python
import synapgrad
from synapgrad import nn, utils

with synapgrad.retain_grads():
    x = synapgrad.tensor([5.0, 3.0], requires_grad=True)
    x2 = synapgrad.tensor([6.0, 0.4], requires_grad=True)
    y = (x ** 3 + 3) 
    y2 = (x2 / x)
    z = y * y2 * x2
    z = z.sum()
    z.backward()
utils.graph.draw(z)
```

![Board Image](/assets/graph_example.svg)

## Running tests
To run the unit tests you will have to install PyTorch. In these tests, gradients calculation as well as losses, layers, etc, are assessed with pytorch to check everything is working fine. To run the tests:
```bash
python -m pytest
```