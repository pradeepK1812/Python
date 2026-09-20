def add(x, y):
    """Add two numbers."""
    return x + y


def subtract(x, y):
    """Subtract y from x."""
    return x - y


def divide(x, y):
    """Divide x by y."""
    if y == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return x / y


def mod_divide(x, y):
    """Perform modular division (x mod y).

    The remainder is always non‑negative, regardless of the sign of ``y``.
    """
    if y == 0:
        raise ZeroDivisionError("Cannot perform modular division by zero")
    # Use the absolute value of the divisor to ensure a positive remainder.
    return x % abs(y)
