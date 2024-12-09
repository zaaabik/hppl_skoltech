import numba
import numpy as np


@numba.njit()
def julia_set_jit(c, lim, img_size, n_iterations):
    x = np.linspace(-lim, lim, img_size)
    y = np.linspace(-lim, lim, img_size)

    real = np.zeros((img_size, img_size))
    imag = np.zeros((img_size, img_size))
    for i in range(img_size):
        for j in range(img_size):
            real[i, j] = x[j]
            imag[i, j] = y[i]

    Z = real + 1j * imag
    iteration_counts = np.zeros(shape=(img_size, img_size), dtype=np.float64)

    for it in range(n_iterations):
        for i in range(img_size):
            for j in range(img_size):
                if abs(Z[i, j]) <= 2:
                    Z[i, j] = Z[i, j] ** 2 + c
                    iteration_counts[i, j] += 1

    return iteration_counts.T
