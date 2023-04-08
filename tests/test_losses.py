import torch
from synapgrad import Tensor, nn
from utils import check_tensors


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
    
    
# def test_BCELoss():
#     ...
    
    
# def test_CrossEntropyLoss():
#     true = [[1.0, 0.0], [0.0,1.0]]
#     pred = [[[1, 0.0], [0.0, 1.0]]]
    
#     # synapgrad
#     y_true = Tensor(true)
#     inp = Tensor(pred, requires_grad=True)
    
#     y_pred = nn.Softmax(dim=1)(inp)
#     y_pred = y_pred.squeeze()
#     loss = nn.CrossEntropyLoss(reduction='mean')(y_pred, y_true)
#     loss.backward()
    
#     print(loss)
    
#     # torch
#     y_true_t = torch.tensor(true)
#     inp_t = torch.tensor(pred, requires_grad=True)
    
#     y_pred_t = torch.nn.Softmax(dim=1)(inp_t)
#     y_pred_t = y_pred_t.squeeze()
#     loss_t = torch.nn.CrossEntropyLoss(reduction='mean')(y_pred_t, y_true_t)
#     loss_t.backward()
    
#     print(loss_t)
    
#     assert check_tensors(loss, loss_t)
#     assert check_tensors(inp.grad, inp_t.grad)