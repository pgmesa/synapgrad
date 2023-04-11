import torch
from synapgrad import Tensor, nn
from utils import check_tensors
import numpy as np


def test_MSELoss():
    true = [1.0, 0]
    pred = [[2.0], [-1.0]]
    
    # synapgrad
    y_true = Tensor(true)
    inp = Tensor(pred, requires_grad=True)
    
    y_pred = nn.Sigmoid()(inp)
    y_pred = y_pred.squeeze()
    loss = nn.MSELoss(reduction='mean')(y_pred, y_true)
    loss.backward()
    
    print(loss)
    
    # torch
    y_true_t = torch.tensor(true)
    inp_t = torch.tensor(pred, requires_grad=True)
    
    y_pred_t = torch.nn.Sigmoid()(inp_t)
    y_pred_t = y_pred_t.squeeze()
    loss_t = torch.nn.MSELoss(reduction='mean')(y_pred_t, y_true_t)
    loss_t.backward()
    
    print(loss_t)
    
    assert check_tensors(loss, loss_t)
    assert check_tensors(inp.grad, inp_t.grad)


def test_NLLLoss():
    pred = [[0.14, 0.3],[-0.2, 0.9],[-3, 0.1]]
    label = [0, 1, 0]
    
    # synapgrad
    ypred = nn.LogSoftmax(dim=1)(Tensor(pred, requires_grad=True))
    ypred.retain_grad()
    ylabel = Tensor(label, dtype=np.int8)
    loss = nn.NLLLoss()(ypred, ylabel)
    loss.backward()

    # torch
    ypred_t = torch.nn.LogSoftmax(dim=1)(torch.tensor(pred, requires_grad=True))
    ypred_t.retain_grad()
    ylabel_t = torch.tensor(label).type(torch.LongTensor)
    loss_t = torch.nn.NLLLoss()(ypred_t, ylabel_t)
    loss_t.backward()

    print(ypred_t.grad)
    print(ypred.grad)
    
    assert check_tensors(loss, loss_t)
    assert check_tensors(ypred.grad, ypred_t.grad)

    
def test_BCELoss():
    pred = [0.3, 0.4, 0.7]
    label = [0, 1, 0]
    
    # synapgrad
    ypred = Tensor(pred, requires_grad=True)
    ylabel = Tensor(label)
    loss = nn.BCELoss()(ypred, ylabel)
    loss.backward()

    # torch
    ypred_t = torch.tensor(pred, requires_grad=True)
    ylabel_t = torch.tensor(label, dtype=torch.float)
    loss_t = torch.nn.BCELoss()(ypred_t, ylabel_t)
    loss_t.backward()

    print(ypred_t.grad)
    print(ypred.grad)
    
    assert check_tensors(loss, loss_t)
    assert check_tensors(ypred.grad, ypred_t.grad)


def test_BCEWithLogitsLoss():
    pred = [0.3, 0.4, 0.7]
    label = [0, 1, 0]
    
    # synapgrad
    ypred = Tensor(pred, requires_grad=True)
    ylabel = Tensor(label)
    loss = nn.BCEWithLogitsLoss()(ypred, ylabel)
    loss.backward()

    # torch
    ypred_t = torch.tensor(pred, requires_grad=True)
    ylabel_t = torch.tensor(label, dtype=torch.float)
    loss_t = torch.nn.BCEWithLogitsLoss()(ypred_t, ylabel_t)
    loss_t.backward()

    print(ypred_t.grad)
    print(ypred.grad)
    
    assert check_tensors(loss, loss_t)
    assert check_tensors(ypred.grad, ypred_t.grad)
    
    
def test_CrossEntropyLoss():
    pred = [[0.14, 0.3],[-0.2, 0.9],[-3, 0.1]]
    label = [0, 1, 0]
    
    # synapgrad
    ypred = Tensor(pred, requires_grad=True)
    ylabel = Tensor(label, dtype=np.int8)
    loss = nn.CrossEntropyLoss()(ypred, ylabel)
    loss.backward()

    # torch
    ypred_t = torch.tensor(pred, requires_grad=True)
    ylabel_t = torch.tensor(label).type(torch.LongTensor)
    loss_t = torch.nn.CrossEntropyLoss()(ypred_t, ylabel_t)
    loss_t.backward()

    print(ypred_t.grad)
    print(ypred.grad)
    
    assert check_tensors(loss, loss_t)
    assert check_tensors(ypred.grad, ypred_t.grad)