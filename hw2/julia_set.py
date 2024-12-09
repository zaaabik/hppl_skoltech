import numpy as np


def julia_set(c, lim, img_size, n_iterations):
    x = np.linspace(-lim, lim, img_size)
    y = np.linspace(-lim, lim, img_size)
    Z = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    Z = Z[:, 0] + 1j * Z[:, 1]

    iteration_counts = np.zeros(Z.shape, dtype=np.int32)

    for i in range(n_iterations):
        mask = np.abs(Z) <= 2
        iteration_counts[mask] += 1
        Z[mask] = Z[mask] ** 2 + c

    return iteration_counts.reshape(img_size, img_size)