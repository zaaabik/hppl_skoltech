import numpy as np

A = -2
B = 2
N = 10 ** 6


def FUNC(x):
    return x ** 2 + 10 + np.sin(x) * 5


def intergral(func, a: float, b: float, n: int):
    dx = (b - a) / n
    result = 0.0
    for i in range(n):
        x = a + i * dx
        result += func(x) * dx
    return result
