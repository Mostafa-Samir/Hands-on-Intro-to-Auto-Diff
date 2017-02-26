from collections import deque
import numpy as np

class Node(np.ndarray):

    def __new__(subtype, shape,
                dtype=float,
                buffer=None,
                offset=0,
                strides=None,
                order=None):
        """
        craetes a new object that wraps an numpy.ndarray into a structure that
        represents a node in a computational graph
        """

        newobj = np.ndarray.__new__(
            subtype, shape, dtype,
            buffer, offset, strides,
            order
        )

        return newobj

    def _nodify(self, method_name, other, opname, self_first=True):
        """
        augments the operation of given arithmetic super method
        by creating and returning an OperationalNode for the operation

        Parameters:
        ----------
        method_name: String
            the name of the super method to be augmented
        other: Node | np.ndarray | Number
            the other operand to the operation
        opname: String
            the name of OperationalNode
        self_first: Boolean
            a flag indicating if self is the 1st operand in non commutative ops

        Returns: OperationalNode
        """
        if not isinstance(other, Node):
            other = ConstantNode.create_using(other)
        opvalue = getattr(np.ndarray, method_name)(self, other)

        return OperationalNode.create_using(opvalue, opname,
            self if self_first else other,
            other if self_first else self
        )


    def __add__(self, other):
        return self._nodify('__add__', other, 'add')

    def __radd__(self, other):
        return self._nodify('__radd__', other, 'add')

    def __sub__(self, other):
        return self._nodify('__sub__', other, 'sub')

    def __rsub__(self, other):
        return self._nodify('__rsub__', other, 'sub', False)

    def __mul__(self, other):
        return self._nodify('__mul__', other, 'mul')

    def __rmul__(self, other):
        return self._nodify('__rmul__', other, 'mul')

    def __div__(self, other):
        return self._nodify('__div__', other, 'div')

    def __rdiv__(self, other):
        return self._nodify('__rdiv__', other, 'div', False)

    def __truediv__(self, other):
        return self._nodify('__truediv__', other, 'div')

    def __rtruediv__(self, other):
        return self._nodify('__rtruediv__', other, 'div', False)

    def __pow__(self, other):
        return self._nodify('__pow__', other, 'pow')

    def __rpow__(self, other):
        return self._nodify('__rpow__', other, 'pow', False)

    @property
    def T(self):
        """
        augments numpy's T attribute by creating a node for the operation
        """
        opvalue = np.transpose(self)
        return OperationalNode.create_using(opvalue, 'transpose', self)


class OperationalNode(Node):

    # a static attribute to count for unnamed nodes
    nodes_counter = {}

    @staticmethod
    def create_using(opresult, opname, operand_a, operand_b=None, name=None):
        """
        craetes an graph node representing an operation

        Parameters:
        ----------
        opresult: np.ndarray
            the result of the operation
        opname: String
            the name of the operation
        operand_a: Node
            the first operand to the operation
        operand_b: Node
            the second operand to the operation if any
        name: String
            the name of the node

        Returns: OperationalNode
        """

        obj = OperationalNode(
            strides=opresult.strides,
            shape=opresult.shape,
            dtype=opresult.dtype,
            buffer=opresult
        )

        obj.opname = opname
        obj.operand_a = operand_a
        obj.operand_b = operand_b

        if name is not None:
            obj.name = name
        else:
            if opname not in OperationalNode.nodes_counter:
                OperationalNode.nodes_counter[opname] = 0

            node_id = OperationalNode.nodes_counter[opname]
            OperationalNode.nodes_counter[opname] += 1
            obj.name = "%s_%d" % (opname, node_id)

        return obj


class ConstantNode(Node):

     # a static attribute to count the unnamed instances
     count = 0

     @staticmethod
     def create_using(val, name=None):
        """
        creates a graph node representing a constant

        Parameters:
        ----------
        val: np.ndarray | Number
         the value of the constant
        name: String
         the node's name
        """
        if not isinstance(val, np.ndarray):
            val = np.array(val, dtype=float)

        obj = ConstantNode(
            strides=val.strides,
            shape=val.shape,
            dtype=val.dtype,
            buffer=val
        )
        if name is not None:
            obj.name = name
        else:
            obj.name = "const_%d" % (ConstantNode.count)
            ConstantNode.count += 1

        return obj


class VariableNode(Node):

     # a static attribute to count the unnamed instances
     count = 0

     @staticmethod
     def create_using(val, name=None):
        """
        creates a graph node representing a variable

        Parameters:
        ----------
        val: np.ndarray | Number
            the value of the constant
        name: String
            the node's name
        """
        if not isinstance(val, np.ndarray):
            val = np.array(val, dtype=float)

        obj = VariableNode(
            strides=val.strides,
            shape=val.shape,
            dtype=val.dtype,
            buffer=val
        )
        if name is not None:
            obj.name = name
        else:
            obj.name = "const_%d" % (VariableNode.count)
            VariableNode.count += 1

        return obj


class NodesQueue:

    def __init__(self):
        """
        creates an object that runs two parallel queus, one for the nodes
        and the other for the node names, this captures the uniqueness of
        a node via name even if it shares the same value as another
        """

        self.nodes = deque()
        self.nodes_ids = deque()

    def push(self, node):
        """
        pushes a given node, along with its name, to the queue

        Parameters:
        ----------
        node: Node
            the node to be pushed
        """
        self.nodes.append(node)
        self.nodes_ids.append(node.name)


    def pop(self):
        """
        pops the front node from the queue, along with its name

        Returns: Node
        """
        node = self.nodes.popleft()
        self.nodes_ids.popleft()

        return node

    def __contains__(self, node):
        """
        implements the searching operator via `in` by searching in the names
        queue instead of the nodes themselves queue to capture unique nodes
        with exact numerical values

        Parameters:
        ----------
        node: Node
            the node to search for
        Returns: Boolean
        """
        return node.name in self.nodes_ids

    def __len__(self):
        """
        returns the length on any of the underlying deques

        Returns: int
        """
        return len(self.nodes)
