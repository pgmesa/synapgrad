from synapgrad.tools import pretty_numpy
from graphviz import Digraph


def trace(root):
    """ root = Tensor """
    nodes, edges = set(), set()
    def build(n):
        if n not in nodes:
            nodes.add(n)
            for child in n._children:
                edges.add((child, n))
                build(child)
    build(root)
    return nodes, edges
    

def draw(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB'], "rankdir must be 'LR' or 'TB'"
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    for n in nodes:
        nid = str(id(n))
        data_str = pretty_numpy(n.data, precision=2)
        grad_str = 'None' if not n.has_grad() else pretty_numpy(n._grad, precision=2)
        grad_fn = 'None' if n.grad_fn is None else n.grad_fn.name()
        header = f"Tensor ({n.name})" if n.name != "" else "Tensor"
        dot.node(name=nid, label=f"<<b>{header}</b> | data={data_str} | req_grad={n.requires_grad}   is_leaf={n.is_leaf} | grad={grad_str} | grad_fn={grad_fn} >", shape='record')
        if n._operation:
            dot.node(name=nid + n._operation, label=n._operation)
            dot.edge(nid + n._operation, nid)
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._operation)
    
    return dot

