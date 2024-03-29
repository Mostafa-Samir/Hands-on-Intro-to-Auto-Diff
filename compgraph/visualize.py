from collections import deque

import networkx as nx
from matplotlib import rc
import matplotlib.pyplot as plt

from compgraph.nodes import *

def visualize_at(node, figsize=None):
    """
    visualizes the graph starting from the given node back to constant and
    variable nodes

    Parameters:
    ----------
    node: nodes.Node
        the node to visualize its computational graph
    """

    G = nx.DiGraph(graph={'rankdir': 'LR'})
    queue = NodesQueue()
    color_dict = {'VariableNode': 'lightblue', 'ConstantNode': 'orange'}
    color = lambda n: color_dict[n.__class__.__name__] if n.__class__.__name__ in color_dict else '#d5a6f9'

    G.add_node(node.name, label=f"${node.name}$", color=color(node))
    queue.push(node)

    while queue:
        current = queue.pop()

        if isinstance(current, VariableNode) or isinstance(current, ConstantNode):
            continue
        
        previous_nodes = sorted(
            filter(lambda n: n is not None, [current.operand_a, current.operand_b]),
            key=lambda n: n.name
        )

        for prev_node in previous_nodes:
            if prev_node is not None:
                G.add_node(prev_node.name, label=f"${prev_node.name}$", color=color(prev_node))
                G.add_edge(prev_node.name, current.name)

                if prev_node not in queue:
                    queue.push(prev_node)

    nodes_colors = [_node[1]["color"] for _node in G.nodes(data=True)]
    nodes_labels = {_node[0]: _node[1]["label"] for _node in G.nodes(data=True)}
    edges_labels = {_edge: '$x$' for _edge in G.edges()}

    pos=nx.nx_pydot.pydot_layout(G, prog='dot')

    rc("mathtext", fontset='cm')
    plt.figure(figsize=figsize)

    nx.draw(G, pos, with_labels=False, arrows=True, node_color=nodes_colors, node_size=2000)
    nx.draw_networkx_labels(G, pos, labels=nodes_labels, font_size=15)
