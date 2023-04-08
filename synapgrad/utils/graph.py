
from graphviz import Digraph


def trace(root):
    nodes, edges = set(), set()
    def build(n):
        if n not in nodes:
            nodes.add(n)
            for child in n._children:
                edges.add((child, n))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    
    for n in nodes:
        nid = str(id(n))
        grad = None if n._grad is None else n._grad.round(decimals=2)
        dot.node(name=nid, label=f"tensor={n.data.round(decimals=2)} | req_grad={n.requires_grad}, is_leaf={n.is_leaf} | grad={grad}", shape='record')
        if n._operation:
            dot.node(name=nid + n._operation, label=n._operation)
            dot.edge(nid + n._operation, nid)
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._operation)
    
    return dot

