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


def max(array, axis=None, name=None):
    """
    defines a node in the computational graph representing a max operation

    Parameters:
    ----------
    array: Node | ndarray | number
        the array to be maxed out
    axis: int
        the axis to perform the max out on
    name: String
        node's name in the graph
    """
    if not isinstance(array, Node):
        array = ConstantNode.create_using(array)
    opvalue = np.max(array, axis=axis)
    opnode = OperationalNode.create_using(opvalue, 'max', array, name=name)

    # save info for gradient computation
    opnode.axis = axis
    opnode.with_keepdims = np.max(array, axis=axis, keep_dims=True)

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
        array_a = ConstantNode.create_using(array_a).resize(condition.shape)
    if not isinstance(array_b, Node):
        array_b = ConstantNode.create_using(array_b).resize(condition.shape)
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
