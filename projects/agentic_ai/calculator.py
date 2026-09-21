"""Simple calculator module providing basic arithmetic operations.

Functions:
- add(a, b): Return the sum of a and b.
- subtract(a, b): Return the difference a - b.
- divide(a, b): Return a / b, raising ZeroDivisionError if b is zero.
- mod_divide(a, b): Return a % b, raising ZeroDivisionError if b is zero.

All functions accept numbers (int or float) and return the appropriate result.
"""

def add(a, b):
    """Return the sum of a and b."""
    return a + b


def subtract(a, b):
    """Return the difference a - b."""
    return a - b


def divide(a, b):
    """Return a divided by b.

    Raises:
        ZeroDivisionError: If b is zero.
    """
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b


def mod_divide(a, b):
    """Return a modulo b (remainder of division).

    Raises:
        ZeroDivisionError: If b is zero.
    """
    if b == 0:
        raise ZeroDivisionError("modulo division by zero")
    return a % b
