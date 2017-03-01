from collections import defaultdict
from compgraph.nodes import *
from visualize import visualize_AD
import numpy as np
import grads

def gradient(node):
    """
    computes and returns the gradient of the given node wrt to VariableNodes
    the function implements a depth-first-search (DFS) to traverse the
    computational graph from the gievn node back to VariableNodes

    Parameters:
    ----------
    node: Node
        the node to compute its gradient
    """

    adjoint = defaultdict(int)
    grad = {}
    queue = NodesQueue()

    # put the given node in the queue and set its adjoint to one
    adjoint[node.name] = ConstantNode.create_using(np.ones(node.shape))
    queue.push(node)

    while len(queue) > 0:
        current_node = queue.pop()

        if isinstance(current_node, ConstantNode):
            continue
        if isinstance(current_node, VariableNode):
            grad[current_node.name] = adjoint[current_node.name]
            continue

        current_adjoint = adjoint[current_node.name]
        current_op = current_node.opname

        op_grad = getattr(grads, '%s_grad' % (current_op))
        next_adjoints = op_grad(current_adjoint, current_node)

        adjoint[current_node.operand_a.name] = adjoint[current_node.operand_a.name] + next_adjoints[0]
        if current_node.operand_a not in queue:
            queue.push(current_node.operand_a)

        if current_node.operand_b is not None:
            adjoint[current_node.operand_b.name] = adjoint[current_node.operand_b.name] + next_adjoints[1]
            if current_node.operand_b not in queue:
                queue.push(current_node.operand_b)

    return grad


def check_gradient(fx, args, suspect):
    """
    checks the correctness of the suspect derivative value against
    the value of the numerical approximation of the derivative

    Parameters:
    ----------
    fx: callable
        The function to check its derivative
    wrt: int
        0-based index of the variable to differntiate with respect to
    args: list
        the values of the function variables at the derivative point
    suspect: float
        the the suspected value of the derivative to check
    """
    h = 1.e-7
    approx_grad = []

    for i in range(len(args)):
        shifted_args = args[:]
        shifted_args[i] = shifted_args[i] + h
        approx_grad.append((fx(*shifted_args) - fx(*args)) / h)

    return np.allclose(approx_grad, suspect)
