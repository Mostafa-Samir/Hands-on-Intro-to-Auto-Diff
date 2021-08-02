import matplotlib.pyplot as plt
from matplotlib import animation, rc
import matplotlib.gridspec as gridspec
import networkx as nx
from compgraph.nodes import *
from collections import defaultdict
import autodiff.grads as grads
import numpy as np

def _sweep_graph(node):
    """
    performs a backward sweep of the graph of the given node to build the nx
    graph object and collect relevant info to the visualization

    Parameters:
    ----------
    node: Node
        the node to sweep its computational graph
    """

    leafs_count = 0
    var_node_names = []
    color_dict = {'VariableNode': 'lightblue', 'ConstantNode': 'orange'}
    color = lambda n: color_dict[n.__class__.__name__] if n.__class__.__name__ in color_dict else 'red'

    queue = NodesQueue()
    G = nx.DiGraph(graph={'rankdir': 'LR'})

    queue.push(node)
    G.add_node(node.name, label=node.name, color=color(node))

    while len(queue) > 0:
        current = queue.pop()
        if isinstance(current, VariableNode) or isinstance(current, ConstantNode):
            if isinstance(current, VariableNode):
                var_node_names.append(current.name)
            if current not in queue:
                leafs_count += 1
            continue
        else:

            previous_nodes = sorted(
                filter(lambda n: n is not None, [current.operand_a, current.operand_b]),
                key=lambda n: n.name
            )

            for prev_node in previous_nodes:
                if prev_node is not None:
                    G.add_node(prev_node.name, label=prev_node.name, color=color(prev_node))
                    G.add_edge(prev_node.name, current.name)

                    if prev_node not in queue:
                        queue.push(prev_node)

    return G, leafs_count, var_node_names


def visualize_AD(node, figsize=None):
    """
    craetes a matplotlib animation visualizing the reverse AD process on the
    the computational graph of the given node

    Parameters:
    ----------
    node: Node
        the node to visualize the reverse AD process on its computational graph
    """

    nx_graph, leafs_count, var_names = _sweep_graph(node)
    frames_count = len(nx_graph.edges()) + leafs_count

    # set the stage for the visualization
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1, hspace=0, wspace=0)

    graph_ax = plt.subplot(gs[0:2, 0])
    chain_ax = plt.subplot(gs[2:3, 0])

    chain_ax.axis("off")
    chain_txt = chain_ax.text(0.2, 0.5, '', fontsize=25, va='center')

    # set the necessary data strutures fro reverse AD
    adjoint = defaultdict(int)
    parameters_dict = {
        'var_names': var_names,
        'nx_graph': nx_graph,
        'adjoint': defaultdict(int), # true if the next call of animate is handeling operand_b
        'queue': NodesQueue(),
        'current_node': None,
        'other_operand': False,
        'grads_annotations': {}
    }
    parameters_dict['adjoint'][node.name] = ConstantNode.create_using(np.ones(node.shape))
    parameters_dict['queue'].push(node)

    def node_grad(node, index):
        """
        returns a string containing the local grad of a node

        Parameters:
        ----------
        node: Node
            the node to print its local grad
        index: int
            the index of the operand to get the grad wrt
        """

        local_grad_txt = ""

        if node.opname == 'add':
            local_grad_txt += "$%s = %s + %s \Rightarrow \\frac{\partial %s}{\partial %s} = 1$" % (
                node.name, node.operand_a.name, node.operand_b.name, node.name,
                node.operand_a.name if index == 0 else node.operand_b.name
            )
        elif node.opname == 'sub':
            local_grad_txt += "$%s = %s - %s \Rightarrow \\frac{\partial %s}{\partial %s} = %s1$" % (
                node.name, node.operand_a.name, node.operand_b.name, node.name,
                node.operand_a.name if index == 0 else node.operand_b.name,
                "" if indx == 0 else "-"
            )
        elif node.opname == 'mul':
            local_grad_txt += "$%s = %s \\times %s \Rightarrow \\frac{\partial %s}{\partial %s} = %s = %.4s$" % (
                node.name, node.operand_a.name, node.operand_b.name, node.name,
                node.operand_a.name if index == 0 else node.operand_b.name,
                node.operand_b.name if index == 0 else node.operand_a.name,
                node.operand_b if index == 0 else node.operand_a
            )
        elif node.opname == 'div':
            local_grad_txt += "$%s = \\frac{%s}{%s} \Rightarrow \\frac{\partial %s}{\partial %s} =" % (
                node.opname, node.operand_a.name, node.operand_b.name, node.name,
                node.operand_a.name if index == 0 else node.operand_b.name
            )
            if index == 0:
                local_grad_txt += "\\frac{1}{%s} = %.4s$" % (node.operand_b.name, 1 / node.operand_b)
            else:
                local_grad_txt += "-\\frac{%s}{%s^2} = %.4s$" % (
                    node.operand_a.name, node.operand_b.name, -1 * node.operand_a / (node.operand_b **2)
                )
        elif node.opname == 'pow':
            local_grad_txt += "$%s = %s^{%s} \Rightarrow \\frac{\partial %s}{\partial %s} =" % (
                node.opname, node.operand_a.name, node.operand_b.name, node.name,
                node.operand_a.name if index == 0 else node.operand_b.name
            )
            if index == 0:
                local_grad_txt += "%s \\times %s^{%s - 1} = %.4s$" % (
                    node.operand_b.name, node.operand_a.name, node.operand_b.name,
                    node.operand_b * (node.operand_a ** (node.operand_b - 1))
                )
            else:
                local_grad_txt += "%s^{%s}\ln %s = %.4s$" % (
                    node.operand_a.name, node.operand_b.name, node.operand_a.name,
                    node * np.log(node.operand_a)
                )
        elif node.opname == 'sin':
            local_grad_txt += "$%s = \sin(%s) \Rightarrow \\frac{\partial %s}{\partial %s} = \cos(%s) = %.4s$" % (
                node.name, node.operand_a.name, node.name, node.operand_a.name, node.operand_a.name,
                np.cos(node.operand_a)
            )
        elif node.opname == 'cos':
            local_grad_txt += "$%s = \cos(%s) \Rightarrow \\frac{\partial %s}{\partial %s} = -\sin(%s) = %.4s$" % (
                node.name, node.operand_a.name, node.name, node.operand_a.name, node.operand_a.name,
                -1 * np.sin(node.operand_a)
            )
        elif node.opname == 'exp':
            local_grad_txt += "$%s = \exp(%s) \Rightarrow \\frac{\partial %s}{\partial %s} = \exp(%s) = %.4s$" % (
                node.name, node.operand_a.name, node.name, node.operand_a.name, node.operand_a.name,
                node
            )
        elif node.opname == 'log':
            local_grad_txt += "$%s = \ln(%s) \Rightarrow \\frac{\partial %s}{\partial %s} = \\frac{1}{%s} = %.4s$" % (
                node.name, node.operand_a.name, node.name, node.operand_a.name, node.operand_a.name,
                1. / node.operand_a
            )
        return local_grad_txt

    def process_edge(current, prev, indx, params):
        """
        performs the necessary processing for an edge between current and prev
        nodes

        Parameters:
        ----------
        current: Node
            the current node passinf it's adjoint
        prev: Node
            the operand node that its adjoint being calculated
        indx: int
            the index of the operand node
        """

        current_adjoint = params['adjoint'][current.name]
        current_op = current.opname

        op_grad = getattr(grads, '%s_grad' % (current_op))
        next_adjoints = op_grad(current_adjoint, current)

        params['adjoint'][prev.name] = params['adjoint'][prev.name] + next_adjoints[indx]

        chain_txt = ""

        if not isinstance(prev, ConstantNode):
            chain_txt += node_grad(current, indx) + "\n"
            chain_txt += "$\\frac{\partial f}{\partial %s} \/ += \/ \\frac{\partial f}{\partial %s}\\frac{\partial %s}{\partial %s} = %.4s\\times%.4s=%.4s$" % (
                    prev.name,
                    current.name,
                    current.name,
                    prev.name,
                    current_adjoint,
                    next_adjoints[indx] / current_adjoint,
                    next_adjoints[indx]
            )

        return chain_txt

    def update_figure(params, chain_txt_buff="", edge_labels={}):
        """
        performs the necessary updates to the axes to craete the frame
        """

        graph_ax.clear()

        node_colors = []
        node_labels = {}
        node_edgecolors = []
        node_edgethickness = []

        for _node in params['nx_graph'].nodes(data=True):
            node_labels[_node[0]] = _node[1]['label']
            if _node[0] == params['current_node'].name:
                node_edgecolors.append("darkgreen")
                node_edgethickness.append(5)
            else:
                node_edgecolors.append(_node[1]['color'])
                node_edgethickness.append(1)
            
            node_colors.append(_node[1]['color'])

        pos=nx.nx_pydot.pydot_layout(params['nx_graph'], prog='dot')

        nx.draw(params['nx_graph'], pos, ax=graph_ax, arrows=True, node_color=node_colors, node_size=2000, edgecolors=node_edgecolors, linewidths=node_edgethickness)
        nx.draw_networkx_labels(params['nx_graph'], pos, ax=graph_ax, labels=node_labels)
        nx.draw_networkx_edge_labels(params['nx_graph'], pos, ax=graph_ax, edge_labels=edge_labels, bbox={'boxstyle':'square,pad=0.1', 'fc':'white', 'ec':'white'}, font_size=20)
        for variable in params['var_names']:
            if variable in params['grads_annotations']:
                params['grads_annotations'][variable].remove()
            node_pos = pos[variable]
            d_txt = "$\\frac{\partial f}{\partial %s} = %.4s$" % (variable, params['adjoint'][variable])
            ant = graph_ax.annotate(d_txt, xy=node_pos, xytext=(-100, 0), textcoords='offset points', size=20, ha='center', va='center')
            params['grads_annotations'][variable] = ant
        chain_txt.set_text(chain_txt_buff)

    def init_func():
        return []

    def animate(i, params):
        """
        sets the content of each frame of the animation
        """

        chain_txt_buff = ""
        edge_labels = {}

        if len(params['queue']) > 0 and not params['other_operand']:
            params['current_node'] = params['queue'].pop()
            current_node = params['current_node']

            if isinstance(current_node, ConstantNode):
                update_figure(params)
                return []
            if isinstance(current_node, VariableNode):
                update_figure(params)
                return []

            chain_txt_buff = process_edge(current_node, current_node.operand_a, 0, params)
            edge_labels[(current_node.operand_a.name, current_node.name)] = "$%.4s$" % (params['adjoint'][current_node.name])

            if current_node.operand_a not in params['queue']:
                params['queue'].push(current_node.operand_a)

            if current_node.operand_b is not None:
                params['other_operand'] = True
        elif len(params['queue']) > 0:
            current_node = params['current_node']

            chain_txt_buff = process_edge(current_node, current_node.operand_b, 1, params)
            edge_labels[(current_node.operand_b.name, current_node.name)] = "$%.4s$" % (params['adjoint'][current_node.name])

            if current_node.operand_b not in params['queue']:
                params['queue'].push(current_node.operand_b)

            params['other_operand'] = False  # reset the flag after processing
        else:
            return []

        update_figure(params, chain_txt_buff, edge_labels)

        return []

    rc('animation', html='html5')
    return animation.FuncAnimation(
        fig, animate, init_func=init_func,
        frames=frames_count, interval=2000,
        fargs=[parameters_dict]
    )
