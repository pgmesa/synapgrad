#  SynapGrad

A lightweight autograd Tensor-based engine with a deep learning library built on top of it made from scratch

[![Downloads](https://static.pepy.tech/personalized-badge/synapgrad?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/synapgrad)


## Installation
|      Requirements       |       Version       |    Version used to develop   |
| :---------------------: | :-----------------: | :--------------------------: |
|     <img src="https://skillicons.dev/icons?i=python&theme=dark" width="48">     |   Only tested in 3.9  |  3.9.16  |
```bash
pip install synapgrad
```
## Technical Description
This project implements a completely functional engine for tracking operations between Tensors, by dynamically building a Directed Acyclic Graph (DAG), and an automatic gradient calculation and backpropagation algorithm (reverse-mode autodiff) over this DAG.

Built on top of the engine, the deep learning library implements the most common functions, layers, losses and optimizers in order to create AI models able to solve real problems.

This library mimics Pytorch in a simplified way, but with similar functions and behaviour. 

## Aim of the project
The objective of this project is to develop a lightweight deep learning library entirely from scratch, without relying on any existing frameworks (such as Keras, PyTorch, TensorFlow, scikit-learn, etc.), using only the `numpy` library.

## Autograd Engine
Automatic gradient calculation and backpropagation algorithm

### Requirements
```r
numpy==1.23.5 # Core
graphviz==0.20.1 # (Optional) Visualize DAG
```

In the `examples/visualize_graph.ipynb` notebook there is an example of how to display the graph that synapgrad creates in the background as operations are chained:

```python
import synapgrad

with synapgrad.retain_grads():
    x1 = synapgrad.tensor([[-5.0, 3.0], [2.0, -4.0]], requires_grad=True)
    x2 = synapgrad.tensor([[6.0, 0.4], [1.9,  2.0]], requires_grad=True)
    x3 = synapgrad.tensor(3.0, requires_grad=True)
    y = synapgrad.addmm(x3, x1, x2) # x3 + x1 @ x2
    z = x2.sqrt() @ y
    z.backward(synapgrad.ones(z.shape))
    
# graphviz is required to draw the graph (pip install graphviz==0.20.1)
z.draw_graph()
```
![Graph Image](/.github/graph_example.svg)

## Deep learning library
Built on top of the engine, synapgrad has a deep learning library that implements the following features:

- `Weight initialization`: Xavier Glorot uniform, Xavier Glorot normal, He Kaiming uniform, He Kaiming normal
- `Activations`: ReLU, LeakyReLU, SELU, Tanh, Sigmoid, Softmax, LogSoftmax
- `Layers`: Linear, Unfold, Fold, Flatten, Dropout
- `Convolutions`: MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d, Conv1d, Conv2d
- `Normalizations`: BatchNorm1d, BatchNorm2d
- `Optimizers`: SGD, Adam, AdamW
- `Losses`: MSELoss, NLLLoss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss

This project includes three Jupyter notebooks (located in `examples/`) that tackle three beginner-level AI problems:

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
# **** torch required if you want to use torch engine instead of synapgrad's *****
torch==1.12.1 # Install following the instructions in https://pytorch.org/
```

### Comparisons with pytorch
To measure the efficiency of synapgrad, all three examples were compared to PyTorch. All training sessions were conducted on a laptop with an Intel Core i7 10th generation processor and 16 GB of RAM.

| Notebook | torch | synapgrad | Model params | Dataset size | Batch size | Epochs |
|     :---:     |  :---:  |  :---:  | :---:  | :---:  | :---:  | :---:  |
| basic_mlp.ipynb | 1.5 s | 1.6 s | 337 | [150, 2] | 4 | 50 |
| mnist_mlp.ipynb | 41 s | 1 min 28 s | 178_710 |  [60_000, 28, 28]  | 64 | 20 | 
| mnist_cnn.ipynb |  2 min 5 s  |  13 min 10 s |  20_586  | [60_000, 1, 28, 28]  |  128  | 5 |

## Running tests

To run the unit tests you will have to install Pytest and PyTorch, which is used to check whether the gradients are calculated correctly
```r
pytest==7.3.1
torch==1.12.1 # Install following the instructions in https://pytorch.org/
```

Run all the tests:
```bash
python -m pytest
```
To compare the speed of `torch` and `synapgrad` in each operation, run the command below:
```
python -m pytest ./tests/test_ops.py -s
```
