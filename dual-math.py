from dualnumbers import DualNumber
from dualnumbers.dmath import *

print "\nDual Numbers Addition/Subtraction:"
x = DualNumber(5.3, 0.65)
y = DualNumber(-3, 1.2)

print "\t (%s) + (%s) = (%s)" % (x, y, x + y)
print "\t (%s) - (%s) = (%s)" % (x, y, x - y)
print "\t (%s) + %s = (%s)" % (y, 10, y + 10)
print "\t %s - (%s) = (%s)" % (5, x, 5 - x)

print "\nDual Numbers Multiplication/Division:"
x = DualNumber(1, 2.36)
y = DualNumber(-6, 5.2)

print "\t (%s) x (%s) = (%s)" % (x, y, x * y)
print "\t (%s) / (%s) = (%s)" % (x, y, x / y)
print "\t (%s) x %s = (%s)" % (y, 3, y * 3)
print "\t %s / (%s) = (%s)" % (5, x, 5 / x)

print "\nPowers Involving Dual Numbers:"
x = DualNumber(1, 2.36)
y = DualNumber(6, 5.2)

print "\t %s ^ (%s) = (%s)" % (5, x, 5 ** x)
print "\t (%s) ^ %s = (%s)" % (y, 2, y ** 2)
print "\t (%s) ^ (%s) = (%s)" % (y, x, y ** x)

print "\nFunctions of Dual Numbers:"
x = DualNumber(2, 1.6)

print "\t log(%s) = %s" % (x, log(x))
print "\t sin(%s) = %s" % (x, sin(x))
print "\t cos(%s) = %s" % (x, cos(x))
print "\t tan(%s) = %s" % (x, tan(x))
print "\t exp(%s) = %s" % (x, exp(x))
print "\t sqrt(%s) = %s" % (x, sqrt(x))
