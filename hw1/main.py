import numpy as np
import rootutils
import plotly.graph_objects as go
import os
root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from hw1.bifurcation_map import x_evolution, get_bifurcation_data

BASE_OUT_FOLDER = os.path.join(root_path, 'hw1', 'out')
DEFAULT_PNG_ARGS = dict(width=1024, height=512, scale=2)


def create_evolution_x_plot():
    r = 3.5
    scatters = []

    xos = np.linspace(0.01, 0.999, 10).tolist()

    for x in xos:
        out = x_evolution(3.5, x, n_iterations=100)
        scatters.append(
            go.Scatter(y=out, name=f'x0={x:.3f}')
        )

    fig = go.Figure(scatters)
    fig.update_layout(title=f'Evolution of X. r={r}', xaxis={'title': 'Step'})
    fig.write_image(os.path.join(BASE_OUT_FOLDER, '1_x_evolution.png'), **DEFAULT_PNG_ARGS)
    fig.write_html(os.path.join(BASE_OUT_FOLDER, '1_x_evolution.html'))


def create_different_r_plot():
    rs = np.linspace(3.5, 4, 5).tolist()
    x0 = 0.7
    n = 200
    m = 100
    scatters = []
    for r in rs:
        out = x_evolution(r, x0, n_iterations=n + m)
        scatters.append(
            go.Scatter(y=out[n:], name=f'r={r}', opacity=0.7)
        )
    os.makedirs(BASE_OUT_FOLDER, exist_ok=True)

    fig = go.Figure(scatters)
    fig.update_layout(title=f'Evolution of X. xo={x0} n={n} m={m}', xaxis={'title': 'Step'})
    fig.write_image(os.path.join(BASE_OUT_FOLDER, '2_different_r_plot.png'), **DEFAULT_PNG_ARGS)
    fig.write_html(os.path.join(BASE_OUT_FOLDER, '2_different_r_plot.html'))


def create_bifurcation_plot():
    r = np.linspace(2.5, 4, 1000).tolist()
    x0 = 0.5
    n = 200
    m = 200
    x, r = get_bifurcation_data(x0, r, n, m)
    fig = go.Figure(go.Scatter(x=r, y=x, mode='markers', marker={'size': 2.5}))
    fig.update_layout(title=f'Bifurcation map', xaxis={'title': 'r'}, yaxis={'title': 'x'})

    fig.write_image(os.path.join(BASE_OUT_FOLDER, '3_bifurcation_plot.png'), width=1024, height=1024, scale=2)


if __name__ == "__main__":
    os.makedirs(BASE_OUT_FOLDER, exist_ok=True)
    # 1. Implement the map, plot the evolution of x (1 point)
    create_evolution_x_plot()

    # 2. Create a linspace of r’s, for every r save the last
    # “m” values of x after the first “n” values (can be m=200, x=200),
    # play around with values (1 point)
    create_different_r_plot()

    # 3. Plot the bifurcation map (1 point)
    create_bifurcation_plot()
