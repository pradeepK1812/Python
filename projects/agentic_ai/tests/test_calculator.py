import pytest
from calculator import add, subtract, divide, mod_divide

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(1, 1) == 0
    assert subtract(0, 5) == -5

def test_divide():
    assert divide(10, 2) == 5
    assert divide(9, 3) == 3
    assert divide(7, 2) == 3.5
    assert divide(-10, 2) == -5
    assert divide(10, -2) == -5
    
    # Test division by zero
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)

def test_mod_divide():
    assert mod_divide(10, 3) == 1
    assert mod_divide(15, 5) == 0
    assert mod_divide(7, 2) == 1
    assert mod_divide(10, -3) == 1  # Negative modulus handled by Python
    
    # Test modular division by zero
    with pytest.raises(ZeroDivisionError):
        mod_divide(10, 0)