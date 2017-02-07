# -*- coding: utf-8 -*-

from __future__ import division
from numbers import Number
from math import log

class DualNumber:

    def __init__(self, real, dual):
        """
        Constructs a dual number: z = real + dual * e: e^2 = 0

        Parameters:
        ----------
        real: Number
            The real component in the dual number
        dual: Number
            The coeffcient of the dual component in the number
        """

        self.real = real
        self.dual = dual

    def _add(self, other):
        """
        Defines addition operation logic for dual numbers

        Parameters:
        ----------
        other: DualNumber | Number
            The other dual/real number to add to the current one
        Returns: DualNumber
            A new dual number containing the addition result
        """
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        elif isinstance(other, Number):
            return DualNumber(self.real + other, self.dual)
        else:
            raise TypeError("Unsupported Type for __add__")

    def _sub(self, other, self_first=True):
        """
        Defines subtractio  operation logic for dual numbers

        Parameters:
        ----------
        other: DualNumber | Number
            The other dual/real number to subtract to the current one
        self_first: Boolean
            An indicator if the current dual is the first operand
            if True, then the operation is (self - other)
            otherwise, the operation is (other - self)
        Returns: DualNumber
            A new dual number containing the subtraction result
        """
        if self_first and isinstance(other, DualNumber):
            return DualNumber(self.real - other.real, self.dual - other.dual)
        elif self_first and isinstance(other, Number):
            return DualNumber(self.real - other, self.dual)
        elif not self_first and isinstance(other, Number):
            return DualNumber(other - self.real, -1 * self.dual)
        else:
            raise TypeError("Unsupported Type for __sub__")

    def _mul(self, other):
        """
        Defines multiplication operation logic for dual numbers

        Parameters:
        -----------
        other: DualNumber | Number
            The other dual/real number to add to multiply
        Returns: DualNumber
            A new dual number containing the multiplication result
        """
        if isinstance(other, DualNumber):
            return DualNumber(self.real * other.real, self.real * other.dual + self.dual * other.real)
        elif isinstance(other, Number):
            return DualNumber(self.real * other, self.dual * other)
        else:
            raise TypeError("Unsupported Type for __mul__")

    def _div(self, other, self_numerator=True):
        """
        Defines division operation logic for dual numbers

        Parameters:
        -----------
        other: DualNumber | Number
            The other dual/real number to add to division
        self_numerator: Boolean
            A flag determining if the current dual is the numerator in the operation
            if True then the operation is (self / other)
            otherwise, other is the base and the operation is (other / self)
        Returns: DualNumber
            A new dual number containing the division result
        """
        if self_numerator and isinstance(other, DualNumber):
            if other.real == 0:
                raise ZeroDivisionError("Attempting to divide by a zero")
            else:
                div_real = self.real / other.real
                div_dual = -1 * (self.real * other.dual - self.dual * other.real) / other.real ** 2
                return DualNumber(div_real, div_dual)
        elif self_numerator and isinstance(other, Number):
            if other == 0:
                raise ZeroDivisionError("Attempting to divide by a zero")
            else:
                return DualNumber(self.real / other, self.dual / other)
        elif not self_numerator and isinstance(other, Number):
            if self.real == 0:
                raise ZeroDivisionError("Attempting to divide by a zero")
            else:
                return DualNumber(other / self.real, -1 * (other * self.dual) / self.real ** 2)
        else:
            raise TypeError("Unsupported Type for __div__")

    def _pow(self, other, self_base=True):
        """
        Defines exponentiation logic for dual numbers

        Parameters:
        -----------
        other: DualNumber | Number
            The exponent of the operation (or the base if self_base is False)
        self_base: Boolean
            A flag determining if the current dual is the base in the operation
            if True then the operation is (self ^ other)
            otherwise, other is the base and the operation is (other ^ self)
        Returns: DualNumber
            A new dual number containing
        """
        if self_base and isinstance(other, Number):
            return DualNumber(self.real ** other, self.dual * other * (self.real ** (other - 1)))
        elif self_base and isinstance(other, DualNumber):
            new_real = self.real ** other.real
            new_dual = (self.real ** (other.real - 1)) * (self.real * other.dual * log(self.real) + other.real * self.dual)
            return DualNumber(new_real, new_dual)
        elif not self_base and isinstance(other, Number):
            return DualNumber(other ** self.real, (other ** self.real) * self.dual * log(other))
        else:
            raise TypeError("Unsupported Type for __pow__")


    def __add__(self, other):
        """
        Overloads the + operator for dual numbers
        """
        return self._add(other)

    def __radd__(self, other):
        """
        Overloads the reverese + operator for dual numbers
        """
        return self._add(other)

    def __sub__(self, other):
        """
        Overloads the - operator for dual numbers
        """
        return self._sub(other)

    def __rsub__(self, other):
        """
        Overloads the reverese - operator for dual numbers
        """
        return self._sub(other, self_first=False)


    def __mul__(self, other):
        """
        Overloads the * operator for dual numbers
        """
        return self._mul(other)

    def __rmul__(self, other):
        """
        Overloads the reverese * operator for dual numbers
        """
        return self._mul(other)

    def __truediv__(self, other):
        """
        Overloads the / operator for dual numbers
        """
        return self._div(other)

    def __rtruediv__(self, other):
        """
        Overloads the reverese / operator for dual numbers
        """
        return self._div(other, self_numerator=False)

    def __div__(self, other):
        """
        Overloads the / operator for dual numbers
        """
        return self._div(other)

    def __rdiv__(self, other):
        """
        Overloads the reverese / operator for dual numbers
        """
        return self._div(other, self_numerator=False)

    def __pow__(self, other):
        """
        Overloads the ** operator for dual numbers
        """
        return self._pow(other)

    def __rpow__(self, other):
        """
        Overloads the reverse ** operator for dual numbers
        """
        return self._pow(other, self_base=False)

    def __cmp__(self, other):
        """
        Overloads comparison operators for dual numbers
        """
        if isinstance(other, DualNumber):
            return self.real - other.real
        elif isinstance(other, Number):
            return self.real - other
        else:
            raise TypeError("Unsupported Type for __cmp__")

    def __repr__(self):
        """
        Provides the string representation of the dual number
        """
        return "%s %s %sɛ" % (self.real, '+' if self.dual > 0 else '-', abs(self.dual))
