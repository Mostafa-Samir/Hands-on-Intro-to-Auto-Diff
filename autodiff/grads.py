import compgraph as cg
import numpy as np

def add_grad(prev_adjoint, node):
    return [prev_adjoint, prev_adjoint]

def sub_grad(prev_adjoint, node):
    return [prev_adjoint, -1 * prev_adjoint]

def mul_grad(prev_adjoint, node):
    return [
        prev_adjoint * node.operand_b,
        prev_adjoint * node.operand_a
    ]

def div_grad(prev_adjoint, node):
    return [
        prev_adjoint / node.operand_b,
        -1 * prev_adjoint * node.operand_a / node.operand_b ** 2
    ]

def pow_grad(prev_adjoint, node):
    return [
        prev_adjoint * node.operand_b * (node.operand_a ** (node.operand_b - 1)),
        prev_adjoint * node * cg.log(node.operand_a)
    ]

def transpose_grad(prev_adjoint, node):
    return [prev_adjoint.T, None]

def sum_grad(prev_adjoint, node):
    return [prev_adjoint * np.ones_like(node.operand_a), None]

def mean_grad(prev_adjoint, node):
    return [prev_adjoint * node * np.ones_like(node.operand_a), None]

def exp_grad(prev_adjoint, node):
    return [prev_adjoint * node, None]

def log_grad(prev_adjoint, node):
    return [prev_adjoint * (1. / node.operand_a), None]

def max_grad(prev_adjoint, node):
    doperand_a = cg.where(node.operand_a == node.with_keepdims, 1, 0)
    normalizers = cg.sum(doperand_a, axis=node.axis, keepdims=True)
    normalized_doperand_a = doperand_a / normalizers

    return [prev_adjoint * normalized_doperand_a, None]

def dot_grad(prev_adjoint, node):
    prev_adj = prev_adjoint
    op_a = node.operand_a
    op_b = node.operand_b

    if prev_adjoint.ndim == 1:
        prev_adj = cg.reshape(prev_adjoint, (1, -1))
    
    if node.operand_b.ndim == 1:
        op_b = cg.reshape(op_b, (-1, 1))

    if node.operand_a.ndim == 1:
        op_a = cg.reshape(op_a, (1, -1))

    return [
        cg.dot(prev_adj, op_b.T),
        cg.dot(op_a.T, prev_adj)
    ]

def where_grad(prev_adjoint, node):
    doperand_a = np.zeros_like(node.operand_a)
    doperand_b = np.ones_like(node.operand_b)

    doperand_a[node.condition] = 1
    doperand_b[node.condition] = 0

    return [prev_adjoint * doperand_a, prev_adjoint * doperand_b]


def sin_grad(prev_adjoint, node):
    return [prev_adjoint * cg.cos(node.operand_a), None]

def cos_grad(prev_adjoint, node):
    return [-1 * prev_adjoint * cg.sin(node.operand_a), None]

def softmax_cross_entropy_grad(prev_adjoint, node):
    return [
        prev_adjoint * (node.softmax_val - node.labels),
        None
    ]

def reshape_grad(prev_adjoint, node):
    return [
        cg.reshape(prev_adjoint, node.operand_a.shape),
        None
    ]

def squeeze_grad(prev_adjoint, node):
    return [
        cg.reshape(prev_adjoint, node.operand_a.shape),
        None
    ]

def unbroadcast_adjoint(node, adjoint):
    """
    puts the adjoint into the correct shape by summing over all the
    brodacsted dimensions. The underlying principle is notthing but
    the multi chain rule.

    Parameters:
    ----------
    node: Node
        the node to check if its adjoint is broadcasted
    adjoint: ndarray
        the the adjoint of the node that might need fixing
    """
    correct_adjoint = adjoint

    if node.shape != adjoint.shape:
        dimensions_diff = np.abs(adjoint.ndim - node.ndim)
        if dimensions_diff != 0:
            summation_dims = tuple(range(dimensions_diff))
            correct_adjoint = cg.sum(adjoint, axis=summation_dims)

            originally_ones = tuple([axis  for axis, size in enumerate(node.shape) if size == 1])
            if len(originally_ones) != 0:
                correct_adjoint = cg.sum(correct_adjoint, axis=axis, keepdims=True)

    return correct_adjoint
