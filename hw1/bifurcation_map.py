def logistic_map(x: float, r: float) -> float:
    return r * x * (1 - x)


def x_evolution(r: float, x0: float, n_iterations=200) -> list:
    x_values = [x0]
    x = x0
    for _ in range(n_iterations):
        x = logistic_map(x, r)
        x_values.append(x)
    return x_values


def get_bifurcation_data(x0, r_values, n, m):
    total_x = []
    total_r = []
    for r in r_values:
        x = x_evolution(r, x0, n + m)
        total_x.extend(x[-m:])
        total_r.extend([r] * m)
    return total_x, total_r
