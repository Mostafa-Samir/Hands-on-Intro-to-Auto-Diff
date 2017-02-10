from dualnumbers import DualNumber

def derivative(fx, wrt, args):
    """
    returns the value of the derivative of the given function with respect to
    the given variable at the point given by the value of args

    Parameters:
    ----------
    fx: callable
        The function to compute its derivative
    wrt: int
        a zero based index of the variable with respect o which the derivative is
        taken
    args: list
        the values of the function variables at the derivative point
    """

    dual_args = []
    for i, arg in enumerate(args):
        if i == wrt:
            dual_args.append(DualNumber(arg, 1))
        else:
            dual_args.append(DualNumber(arg, 0))

    return fx(*dual_args).dual


def gradient(fx, args):
    """
    returns the gradient of a function at the given point by values of args

    Parameters:
    ----------
    fx: callable
        the function to compuet its gradient
    args: list
        the values of the function's variables at the gradient point
    """

    grad = []
    for i,_ in enumerate(args):
        grad.append(derivative(fx, i, args))

    return grad


def check_derivative(fx, wrt, args, suspect):
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
    rerr = 1.e-5
    aerr = 1.e-8

    shifted_args = args[:]
    shifted_args[wrt] += h

    numerical_derivative = (fx(*shifted_args) - fx(*args)) / h
    accept_error = aerr + abs(numerical_derivative) * rerr
    return abs(suspect - numerical_derivative) <= accept_error


def differntiate(fx, wrt):
    """
    returns a callable that evaluates the first derivative (possibly partial)
    of the given function `fx` with respect to a given variable

    Parameters:
    ----------
    fx: callable
        The function to differntiate
    wrt: int
        0-based index of the variable to differntiate with respect to
    """

    def dfx(*values):
        """
        This is the callable derivative
        """

        dual_values = []
        for i, val in enumerate(values):
            if wrt == i:
                dual_values.append(DualNumber(val, 1))
            else:
                dual_values.append(DualNumber(val, 0))

        return fx(*dual_values).dual

    return dfx
