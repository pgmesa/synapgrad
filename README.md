#  SynapGrad

This is another implementation of synapgrad (not available in PyPI at the moment), but with a more Torch-like engine. It is a better-documented and structured project that additionally includes new functionalities. It is not in the main branch because, currently, this implementation, despite being, in my opinion, a better project, is somewhat slower than my previous implementation. 

I need to investigate the cause of this slowdown. The backward pass is specially slow compared to my first implementation. If I manage to identify the bottleneck and fix it, I will update the main branch with this implementation.

If someone discovers the issue causing the slowdown and knows how to fix it, please don't hesitate to create a pull request or contact me.

## Autograd
Automatic gradient calculation and backpropagation algorithm

### Requirements
```r
numpy==1.23.5 # Core
graphviz==0.20.1 # Visualize DAG
```

In the `examples/visualize_graph.ipynb` notebook there is an example of how to display the graph that synapgrad creates in the background as operations are chained.

```python
import synapgrad
from synapgrad import nn

with synapgrad.retain_all():
    x1 = synapgrad.tensor([[-5.0, 3.0], [2.0, -4.0]], requires_grad=True)
    x2 = synapgrad.tensor([[6.0, 0.4], [1.9,  2.0]], requires_grad=True)
    x3 = synapgrad.tensor(3.0, requires_grad=True)
    y = synapgrad.addmm(x3, x1, x2)
    z = x2.sqrt() @ y
    z.backward(synapgrad.ones(z.shape))
z.draw_graph()
```
![Graph Image](/.github/graph_example.svg)

## Deep learning library
Built on top of the engine, synapgrad has a deep learning library that implements the following features:

- `Weight initialization`: Xavier Glorot uniform, Xavier Glorot normal, He Kaiming uniform, He Kaiming normal, LeCun uniform
- `Activations`: ReLU, Tanh, Sigmoid, Softmax, LogSoftmax
- `Convolutions`: MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d, Conv1d, Conv2d
- `Layers`: Linear, Unfold, Fold, BatchNorm1d, BatchNorm2d, Flatten, Dropout
- `Optimizers`: SGD, Adam
- `Losses`: MSELoss, NLLLoss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss


This project comes with 3 jupyter notebooks (in `examples/`) that solve 3 beginner's problems in AI:

- [x] 1. Basic MLP for binary classification (sklearn 'make_moons' toy dataset)
- [x] 2. MLP for handwritten digits classification (MNIST dataset) 
- [x] 3. CNN for handwritten digits classification (MNIST dataset)

### Notebook requirements
```r
pkbar==0.5
matplotlib==3.7.0
ipykernel==6.19.2
scikit-learn==1.2.1
torchvision==0.13.1
# **** Also need it to run the tests *****
torch==1.12.1 # Install following the instructions in https://pytorch.org/
```

## Running tests
To run the unit tests you will have to install PyTorch, which is used to check whether the gradients are calculated correctly
```bash
python -m pytest
```
To compare the speed of `torch` and `synapgrad` in each operation, run the command below:
```
pytest ./tests/test_ops.py -s
```