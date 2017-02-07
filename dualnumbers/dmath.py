from dualnumbers import DualNumber
import math as rmath

def log(x):
    """
    Extends the log operation to dual numbers

    Parameters:
    ----------
    x: DualNumber | Number
        The log operand
    """
    if isinstance(x, DualNumber):
        return DualNumber(rmath.log(x.real), x.dual / x.real)
    else:
        return rmath.log(x)


def sin(x):
    """
    Extends the sin operation to dual numbers

    Parameters:
    ----------
    x: DualNumber | Number
        The sin operand
    """
    if isinstance(x, DualNumber):
        return DualNumber(rmath.sin(x.real), rmath.cos(x.real) * x.dual)
    else:
        return rmath.sin(x)

def cos(x):
    """
    Extends the cos operation to dual numbers

    Parameters:
    ----------
    x: DualNumber | Number
        The cos operand
    """
    if isinstance(x, DualNumber):
        return DualNumber(rmath.cos(x.real), -1 * rmath.sin(x.real) * x.dual)
    else:
        return rmath.cos(x)


def tan(x):
    """
    Extends the tan operation to dual numbers

    Parameters:
    ----------
    x: DualNumber | Number
        The tan operand
    """
    if isinstance(x, DualNumber):
        return DualNumber(rmath.tan(x.real), x.dual * (1 / rmath.cos(x.real)) ** 2)
    else:
        return rmath.tan(x)

def exp(x):
    """
    Extends the exp operation to dual numbers

    Parameters:
    ----------
    x: DualNumber | Number
        The exp operand
    """
    if isinstance(x, DualNumber):
        return DualNumber(rmath.exp(x.real), rmath.exp(x.real) * x.dual)
    else:
        return rmath.exp(x)

def sqrt(x):
    """
    Extends the sqrt operation to dual numbers

    Parameters:
    ----------
    x: DualNumber | Number
        The sqrt operand
    """
    if isinstance(x, DualNumber):
        return DualNumber(rmath.sqrt(x.real), (0.5 / rmath.sqrt(x.real)) * x.dual)
    else:
        return rmath.sqrt(x)
