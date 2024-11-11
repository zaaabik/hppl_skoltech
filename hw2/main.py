import numpy as np
import plotly.graph_objects as go
import rootutils
import os
import imageio
from imageio.v3 import imread

root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from hw2.julia_set import julia_set


BASE_OUT_FOLDER = os.path.join(root_path, 'hw2', 'out')
DEFAULT_PNG_ARGS = dict(width=512, height=512, scale=2)

C = [
    -0.744 + 0.148j,
    -0.741 + 0.143j,
    0.285 + 0.01j
]


def black_and_white_plot():
    lim = 2
    img_size = 1024
    n_iterations = 100

    for c in C:
        Z = julia_set(c, lim, img_size, n_iterations)
        Z = (Z != n_iterations).astype(float)
        fig = go.Figure(data=go.Heatmap(z=Z, colorscale='gray', showscale=False))
        fig.update_layout(title=f'Black and white C={c}',
                          xaxis={'title': 'Re(z0)'}, yaxis={'title': 'Im(z0)'})
        fig.write_image(os.path.join(BASE_OUT_FOLDER, f'1_black_and_white_plot_{c}.png'), **DEFAULT_PNG_ARGS)


def color_plot():
    lim = 2
    img_size = 1024
    n_iterations = 512

    for c in C:
        Z = julia_set(c, lim, img_size, n_iterations)
        fig = go.Figure(data=go.Heatmap(z=Z, colorscale='rainbow', showscale=False))
        fig.update_layout(title=f'Black and white C={c}',
                          xaxis={'title': 'Re(z0)'}, yaxis={'title': 'Im(z0)'})
        fig.write_image(os.path.join(BASE_OUT_FOLDER, f'2_color_plot_{c}.png'), **DEFAULT_PNG_ARGS)


def golden_ratio_plot():
    lim = 2
    img_size = 1024
    n_iterations = 512
    c = 1 - (1 + np.sqrt(5)) / 2

    Z = julia_set(
        c, lim,
        img_size, n_iterations
    )
    fig = go.Figure(data=go.Heatmap(z=Z, showscale=False, zmin=0, zmax=n_iterations))
    fig.update_layout(title=f'Golden ratio C={c:.3f}',
                      xaxis={'title': 'Re(z0)'}, yaxis={'title': 'Im(z0)'})
    fig.write_image(os.path.join(BASE_OUT_FOLDER, f'3_golden_ratio_plot_{c:.3f}.png'), **DEFAULT_PNG_ARGS)


def animation():
    lim = 2
    img_size = 1024
    n_iterations = 512
    C = np.exp(np.linspace(0, 2 * np.pi, 5) * 1j)
    for idx, c in enumerate(C):
        Z = julia_set(c, lim, img_size, n_iterations)
        fig = go.Figure(data=go.Heatmap(z=Z, showscale=False, zmin=0, zmax=n_iterations))
        fig.update_layout(title=f" Julia's set ={c}",
                          xaxis={'title': 'Re(z0)'}, yaxis={'title': 'Im(z0)'})
        fig.write_image(os.path.join(BASE_OUT_FOLDER, f'4_animation_{c}_{idx}.png'), **DEFAULT_PNG_ARGS)


def plot_and_save_julia_set_gif():
    frames = []
    a_values = np.linspace(0, 2 * np.pi, 100)

    lim = 2
    img_size = 512
    n_iterations = 100

    for a in a_values:
        c = np.exp(1j * a)
        iterations = julia_set(c, lim, img_size, n_iterations)

        fig = go.Figure(data=go.Heatmap(z=iterations, colorscale='Viridis', zmin=0, zmax=n_iterations, showscale=False))
        fig.update_layout(title=f'Julia Set for c={c:.2f}', xaxis_title='Re(z0)', yaxis_title='Im(z0)')

        img_filename = f'frame_{a:.2f}.png'
        fig.write_image(img_filename)
        frames.append(img_filename)

    # Create GIF from saved images
    with imageio.get_writer(os.path.join(BASE_OUT_FOLDER, 'animation.gif'), mode='I', duration=0.5) as writer:
        for frame in frames:
            image = imread(frame)
            writer.append_data(image)

    # Clean up: Remove individual frame images
    for frame in frames:
        os.remove(frame)


if __name__ == "__main__":
    os.makedirs(BASE_OUT_FOLDER, exist_ok=True)

    # 1. Make a two color plot, e.g black – the value of z converges, white – diverges (1 point)
    black_and_white_plot()

    # 2. Use more than two colors for bifurcation points
    # (you can also create your own coloring logic or look for proposals
    # on the internet or use the one provided in the step-by-step guide) (1 point)
    color_plot()

    # 3. Generate figure of Julia set (c = 1-r) where r
    # is the golden ratio. Label the axes Re(z0) and Im(z0) (2 points)
    golden_ratio_plot()

    # 4. Plot the figures for c=exp(ia), a = range(0,2pi) and write down the axes
    # like in subtask 3, create animation of
    # these figures slowly changing the value of a (3 points)
    # animation()
    plot_and_save_julia_set_gif()

