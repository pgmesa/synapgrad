# A deep learning library from scratch

This project is based on the amazing educational project 'micrograd' from 'karpathy' (https://github.com/karpathy/micrograd)

## Technical Description
This project implements a completely functional engine for tracking operations between Tensors, by dynamically building a Directed Acyclic Graph (DAG), and an automatic backpropagation algorithm (reverse-mode autodiff) over this DAG. 

Built on top of the engine, the deep learning library implements the most common functions, layers, losses and optimizers in order to create MLPs and CNNs able to solve basic AI problems in a reasonable time

This library tries to mimic Pytorch in a very simplified way, but with similar functions and behaviour. 

## Aim of the project
The aim of this project is to create a deep learning library from scratch, without using any existing framework such as keras, pytorch, tensorflow, sklearn, etc. However, some of these are used for the following reasons:

- (pytorch) to check gradient calculation is correct
- (sklearn) (keras) to download the example datasets

Note: Supporting GPU execution is out of the scope of this project

## Installation
```
pip install deeplib
```

## Autograd Example
```
import deeplib as dl

a = dl.tensor([2.0, 4.0], requieres_grad=True)
```

## Training examples using deeplib

This project comes with 3 jupyter notebooks that solve 3 beginner's problems in AI:
1. Basic MLP for binary classification (sklearn 'make_moons' toy dataset)
2. MLP for handwritten digits classification (MNIST dataset) 
3. CNN for handwritten digits classification (MNIST dataset)

Example 1 (deeplib MLP solution)     |  Example 2 and 3
:-------------------------:|:-------------------------:
![Board Image](/assets/example1.png) | ![Check Image](/assets/example23.png) 

## Comparisons with other frameworks
In order to see the efficiency of deeplib, it is compared with other existing engines (pytorch and micrograd).


| Training Example | deeplib | pytorch | micrograd |
|     :---:        |  :---:  |  :---:  |   :---:   |  
| 1  | 33.2 s | 1.5 s | 1 min y 43 s |
| 2  | - | - | - |
| 3  | - | - | - |

As it was expected, deeplib is faster than micrograd but much slower that pytorch.

## Graph Visualization

## Running tests
```
python -m pytest
```