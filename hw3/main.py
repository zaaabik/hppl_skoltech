import plotly.graph_objects as go
import rootutils
import os

root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from utils.image import write_gif_from_figures
from hw3.schelling_model import initialize_grid, simulate_step

BASE_OUT_FOLDER = os.path.join(root_path, 'hw3', 'out')
DEFAULT_PNG_ARGS = dict(width=512, height=512, scale=2)


def plot_schelling_animation(grid_size: int, r: float, iterations: int) -> list[int]:
    initial_grid = initialize_grid(grid_size)
    grid = initial_grid
    num_dissatisfied_agents_during_sim = []
    data = []

    for iteration in range(iterations):
        grid, num_dissatisfied_agents = simulate_step(grid, r)
        num_dissatisfied_agents_during_sim.append(num_dissatisfied_agents)
        data.append(go.Heatmap(z=grid, colorscale=['white', 'blue', 'red'], showscale=False))

    fig = go.Figure(
        data=[go.Heatmap(z=initial_grid, colorscale=['white', 'blue', 'red'], showscale=False)],
        layout=go.Layout(
            title=f"Schelling's Model Animation (R = {r})",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        ),
        frames=[
            go.Frame(data=data, name=str(iteration)) for data, iteration in zip(data, range(iterations))
        ]
    )
    file_name = f'schelling_model_r={r:.3f}'
    fig.write_html(os.path.join(BASE_OUT_FOLDER, file_name + '.html'))
    write_gif_from_figures(
        [
            go.Figure(d, layout_title=f'R={r} Iteration={iteration}') for d, iteration in zip(data, range(iterations))
        ], os.path.join(BASE_OUT_FOLDER, file_name + '.gif')
    )
    return num_dissatisfied_agents_during_sim


def plot_dissatisfied_agents(dissatisfied_agents: list[list[int]], r_values: list[int]):
    assert len(r_values) == len(dissatisfied_agents), 'Len of r values should be equal to len of dissatisfied_agents'
    scatters = []
    for r, dissatisfied_agents_for_r in zip(r_values, dissatisfied_agents):
        scatters.append(
            go.Scatter(y=dissatisfied_agents_for_r, name=f'R={r}')
        )
    fig = go.Figure(scatters, layout_title='Number of dissatisfied agents during simulation')
    fig.update_xaxes(title='Iteration')
    fig.update_yaxes(title='Number of dissatisfied agents')
    fig.write_image(os.path.join(BASE_OUT_FOLDER, 'num_of_dissatisfied_agents.png'))


if __name__ == '__main__':
    os.makedirs(BASE_OUT_FOLDER, exist_ok=True)

    grid_size = 50

    r_values = [
        i / 8 for i in range(0, 9)
    ]
    num_iterations = 50
    # 1. Create 9 gifs of map evolution for 9 values of R (5 points)
    num_dissatisfied_agents = []
    for r in r_values:
        num_dissatisfied_agents.append(
            plot_schelling_animation(
                grid_size, r=r, iterations=num_iterations)
        )

    # 2. Plot the number of households that want to move versus time for 9 values of R on one graph,
    # label 9 curves, label the axes and title the graph. (2 points)
    plot_dissatisfied_agents(num_dissatisfied_agents, r_values)
