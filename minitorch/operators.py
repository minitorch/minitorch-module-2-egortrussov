"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Any

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -1.0 * x


def lt(x: float, y: float) -> float:
    # return float(x < y)
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    # return float(x == y)
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    return x if x >= y else y


def is_close(x: float, y: float) -> float:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x))


def relu(x: float) -> float:
    return (x > 0) * x


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    return 1 / x


def log_back(x: float, y: float) -> Any:
    return y / x


def inv_back(x: float, y: float) -> Any:
    # return neg(y) / x ** 2
    return -y / x**2


def relu_back(x: float, y: float) -> Any:
    return (x > 0) * y


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(f: Any) -> Any:
    def wrapper(x: Any) -> Any:
        return [f(i) for i in x]

    return wrapper


def zipWith(f: Any) -> Any:
    def wrapper(x: Any, y: Any) -> Any:
        return [f(a, b) for a, b in zip(x, y)]

    return wrapper


def reduce(f: Any, initializer: Any) -> Any:
    def wrapper(x: Any) -> Any:
        val = initializer
        for i in x:
            val = f(val, i)
        return val

    return wrapper


def negList(x: Any) -> Any:
    return map(neg)(x)


def addLists(x: Any, y: Any) -> Any:
    return zipWith(add)(x, y)


def sum(x: Any) -> Any:
    return reduce(add, 0)(x)


def prod(x: Any) -> Any:
    return reduce(mul, 1)(x)
