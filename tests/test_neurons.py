import torch
from synapgrad import Tensor, nn
from utils import check_tensors


def test_neuron():
    list_ = [[0.2, 1, 4, 2, -1, 4], [0.2, 1, 4, 2, -1, 4]]
    
    # synapgrad
    inp = Tensor(list_, requires_grad=True)
    neuron = nn.Neuron(6)
    out = neuron(inp)
    out = out.sum()
    out.backward()

    # torch
    inp_t = torch.tensor(list_, requires_grad=True)
    neuron_t = torch.nn.Linear(6,1)
    out_t = neuron_t(inp_t)
    out_t = out_t.sum()
    out_t.backward()

    params = neuron.parameters()
    params_t = list(neuron_t.parameters())

    assert len(params) == len(params_t)
    for p, p_t in zip(params, params_t):
        assert check_tensors(p.grad, p_t.grad)
        