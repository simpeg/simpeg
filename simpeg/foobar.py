def f(x):
    # black should complain about the spaces between the pow operator
    return x ** 2


def g(x, y):
    # flake8 should complain about unused variable
    z = x == y
    return x + y
