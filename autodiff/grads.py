import compgraph as cg

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

def dot(prev_adjoint, node):
    return [
        cg.dot(prev_adjoint, node.operand_b.T),
        cg.dot(node.operand_a.T, prev_adjoint)
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
