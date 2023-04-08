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
    
    # torch
    y_true_t = torch.tensor(true)
    inp_t = torch.tensor(pred, requires_grad=True)
    
    y_pred_t = torch.nn.Sigmoid()(inp_t)
    y_pred_t = y_pred_t.squeeze()
    loss_t = torch.nn.MSELoss(reduction='mean')(y_pred_t, y_true_t)
    loss_t.backward()
    
    assert check_tensors(inp.grad, inp_t.grad)