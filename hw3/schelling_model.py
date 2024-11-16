from random import shuffle
import numpy as np


def initialize_grid(size: int, p_empty: float = 0.2, p_agent: float = 0.4):
    assert np.isclose(p_empty + p_agent * 2, 1.)

    grid = np.random.choice(
        a=(0, 1, 2),
        size=(size, size),
        p=(p_empty, p_agent, p_agent)
    )
    return grid


def get_neighbors(grid: np.ndarray, row: int, col: int):
    neighbors = []
    rows, cols = grid.shape
    for i in range(max(0, row - 1), min(rows, row + 2)):
        for j in range(max(0, col - 1), min(cols, col + 2)):
            if (i != row or j != col) and grid[i, j] != 0:
                neighbors.append(grid[i, j])
    return neighbors


def is_satisfied(grid: np.ndarray, row: int, col: int, r: float):
    neighbors = get_neighbors(grid, row, col)
    if not neighbors:
        return True
    agent = grid[row, col]
    same_type_count = neighbors.count(agent)
    satisfaction_ratio = same_type_count / len(neighbors)
    return satisfaction_ratio >= r


def simulate_step(grid: np.ndarray, r: float):
    grid = grid.copy()
    dissatisfied_agents = []
    rows, cols = grid.shape
    empty_cells = list(zip(*np.where(grid == 0)))

    for row in range(rows):
        for col in range(cols):
            if grid[row, col] != 0 and not is_satisfied(grid, row, col, r):
                dissatisfied_agents.append((row, col))

    shuffle(dissatisfied_agents)
    shuffle(empty_cells)

    num_dissatisfied_agents = len(dissatisfied_agents)
    for empty_cells, dissatisfied_agents in zip(empty_cells, dissatisfied_agents):
        grid[*empty_cells] = grid[dissatisfied_agents[0], dissatisfied_agents[1]]
        grid[dissatisfied_agents[0], dissatisfied_agents[1]] = 0

    return grid, num_dissatisfied_agents
