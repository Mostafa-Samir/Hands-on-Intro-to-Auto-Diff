import numpy as np
from nodes import *

def variable(initial_value, name=None):
    """
    defines a node in the computational graph representing a variable

    Parameters:
    ----------
    initial_value: np.ndarray | Number
        the initial value of the variable
    name: String
        the name of the variable node
    """
    return VariableNode.create_using(initial_value, name)


def constant(value, name=None):
    """
    defines a node in the computational graph representing a constant

    Parameters:
    ----------
    value: np.ndarray | Number
        initial value of the constant
    name: String
        the name of the constant node
    """
    return ConstantNode.create_using(value, name)


def sum(array, axis=None, keepdims=False, name=None):
    """
    defines a node in the computational graph representing a sum operation

    Parameters:
    ----------
    array: Node | ndarray | number
        the array to be summed
    axis: int
        the axis to perform the sum on
    keepdims: Boolean
        a flag to determine if the dimensions are kept
    name: String
        node's name in the graph
    """
    if not isinstance(array, Node):
        array = ConstantNode.create_using(array)
    opvalue = np.sum(array, axis=axis, keepdims=keepdims)

    return OperationalNode.create_using(opvalue, 'sum', array, name=name)


def mean(array, axis=None, name=None):
    """
    defines a node in the computational graph representing a mean operation

    Parameters:
    ----------
    array: Node | ndarray | number
        the array to be averaged
    axis: int
        the axis to perform the averaging on
    name: String
        node's name in the graph
    """
    if not isinstance(array, Node):
        array = ConstantNode.create_using(array)
    opvalue = np.mean(array, axis=axis)

    return OperationalNode.create_using(opvalue, 'mean', array, name=name)


def exp(array, name=None):
    """
    defines a node in the computational graph representing an exp operation

    Parameters:
    ----------
    array: Node | ndarray | number
        the array to be exp-ed
    name: String
        node's name in the graph
    """
    if not isinstance(array, Node):
        array = ConstantNode.create_using(array)
    opvalue = np.exp(array)

    return OperationalNode.create_using(opvalue, 'exp', array, name=name)


def log(array, name=None):
    """
    defines a node in the computational graph representing an log operation

    Parameters:
    ----------
    array: Node | ndarray | number
        the array to be log-ed
    name: String
        node's name in the graph
    """
    if not isinstance(array, Node):
        array = ConstantNode.create_using(array)
    opvalue = np.log(array)

    return OperationalNode.create_using(opvalue, 'log', array, name=name)


def max(array, axis=None, keepdims=False, name=None):
    """
    defines a node in the computational graph representing a max operation

    Parameters:
    ----------
    array: Node | ndarray | number
        the array to be maxed out
    axis: int
        the axis to perform the max out on
    keepdims: Boolean
        a flag to determine if the dimensions are kept
    name: String
        node's name in the graph
    """
    if not isinstance(array, Node):
        array = ConstantNode.create_using(array)
    opvalue = np.max(array, axis=axis, keepdims=keepdims)
    opnode = OperationalNode.create_using(opvalue, 'max', array, name=name)

    # save info for gradient computation
    opnode.axis = axis
    opnode.keepdims = keepdims
    opnode.with_keepdims = np.max(array, axis=axis, keepdims=True)

    return opnode


def dot(array_a, array_b, name=None):
    """
    defines a node in the computational graph representing an array product op

    Parameters:
    ----------
    array_a: Node | ndarray | number
        the first operand to the product
    array_b: Node | ndarray | number
        the second operand to the product
    name: String
        the name of the node
    """
    if not isinstance(array_a, Node):
        array_a = ConstantNode.create_using(array_a)
    if not isinstance(array_b, Node):
        array_b = ConstantNode.create_using(array_b)
    opvalue = np.dot(array_a, array_b)

    return OperationalNode.create_using(opvalue, 'dot', array_a, array_b, name)


def where(condition, array_a, array_b, name=None):
    """
    defines a node in the computational graph representing a where selection
    operation

    Parameters:
    ----------
    condition: ndarray of Boolean
        the selection condition
    array_a: Node | ndarray | number
        the value to select from when the condition is True
    array_b: Node | ndarray | number
        the value to select from when the condition is False
    name: String
        the name of the node
    """
    if not isinstance(array_a, Node):
        nd_array_a = np.full_like(condition, array_a)
        array_a = ConstantNode.create_using(nd_array_a)
    if not isinstance(array_b, Node):
        nd_array_b = np.full_like(condition, array_b)
        array_b = ConstantNode.create_using(nd_array_b)
    opvalue = np.where(condition, array_a, array_b)
    opnode = OperationalNode.create_using(opvalue, 'where', array_a, array_b, name=name)
    opnode.condition = condition  # save condition for gradient computation

    return opnode


def sin(array, name=None):
    """
    defines a node in the computational graph representing a sin operation

    Parameters:
    ----------
    array: Node | ndarray | number
        the array to be sin-ed
    name: String
        node's name in the graph
    """
    if not isinstance(array, Node):
        array = ConstantNode.create_using(array)
    opvalue = np.sin(array)

    return OperationalNode.create_using(opvalue, 'sin', array, name=name)

def cos(array, name=None):
    """
    defines a node in the computational graph representing a cos operation

    Parameters:
    ----------
    array: Node | ndarray | number
        the array to be sin-ed
    name: String
        node's name in the graph
    """
    if not isinstance(array, Node):
        array = ConstantNode.create_using(array)
    opvalue = np.cos(array)

    return OperationalNode.create_using(opvalue, 'cos', array, name=name)

def softmax_cross_entropy(logits, labels, name=None):
    """
    defines a softmax-cross-entropy op as a primitive for numerical stability

    Parameters:
    ----------
    logits: Node| ndarray| Number
        the model's prediction
    labels:
        the true labels
    name: String
        node's name in the graph
    """
    if not isinstance(logits, Node):
        logits = ConstantNode.create_using(logits)
    if not isinstance(labels, Node):
        labels = ConstantNode.create_using(labels)

    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_op = np.exp(logits - logits_max)
    logits_softmax = exp_op / np.sum(exp_op, axis=1, keepdims=True)

    cross_entropy = -1 * np.mean(labels * np.log(logits_softmax + 1e-7))

    opnode = OperationalNode.create_using(
        cross_entropy,
        'softmax_cross_entropy',
        logits,
        name=name
    )

    # save info for gradient calculations
    opnode.softmax_val = logits_softmax
    opnode.labels = labels

    return opnode

def reshape(array, new_shape, name=None):
    """
    defines a node in the computational graph representing a reshape operation

    Parameters:
    ----------
    array: Node| ndarray
        the array to be reshaped
    new_shape: iterable
        the new shape to put the array in
    name: String
        node's name in the graph
    """
    if not isinstance(array, Node):
        array = ConstantNode.create_using(array)
    opvalue = np.reshape(array, new_shape)

    return OperationalNode.create_using(opvalue, 'reshape', array, name=name)

def squeeze(array, axis=None, name=None):
    """
    defines a node in the computational graph representing a squeeze operation

    Parameters:
    ----------
    array: Node| ndarray
        the array to be squeezed
    axis: iterable
        the 1 axes to be squeezed out of the array
    name: String
        node's name in the graph
    """
    if not isinstance(array, Node):
        array = ConstantNode.create_using(array)
    opvalue = np.squeeze(array, axis=axis)

    return OperationalNode.create_using(opvalue, 'squeeze', array, name=name)
