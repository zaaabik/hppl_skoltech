import cProfile
import os
import pstats
import timeit

import numpy as np
import rootutils
from PIL import Image
from plotly import graph_objs as go

root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
BASE_OUT_FOLDER = os.path.join(root_path, 'hw8', 'out')

from hw8.julia_set_jit import julia_set_jit
from hw2.julia_set import julia_set

DEFAULT_SQUARE_PNG_ARGS = dict(width=1024, height=1024, scale=2)
DEFAULT_PNG_ARGS = dict(width=1024, height=512, scale=2)


def compare_speed():
    lim = 2
    img_size = 16
    n_iterations = 512
    c = 1 - (1 + np.sqrt(5)) / 2

    jit_speed = np.array(timeit.repeat(lambda:
                         julia_set_jit(
                             c, lim,
                             img_size, n_iterations
                         ),
                         number=1, repeat=1))

    python_speed = np.array(timeit.repeat(lambda:
                         julia_set(
                             c, lim,
                             img_size, n_iterations
                         ),
                         number=1, repeat=1))
    fig = go.Figure(go.Bar(
        x=['Python implementation', 'JIT'],
        y=[python_speed.mean(), jit_speed.mean()],
        error_y=dict(type='data', array=[python_speed.std(), jit_speed.std()])
    ))

    fig.update_yaxes(title='Speed (s)')
    fig.update_layout(title='Mandelbrot set JIT speedup')
    fig.write_image(os.path.join(BASE_OUT_FOLDER, 'JIT_speedup.png'), **DEFAULT_PNG_ARGS)


if __name__ == '__main__':
    os.makedirs(BASE_OUT_FOLDER, exist_ok=True)
    lim = 2
    img_size = 1024
    n_iterations = 512
    c = 1 - (1 + np.sqrt(5)) / 2

    julia_set_jit(c, lim, img_size, n_iterations)
    original_res = julia_set_jit(c, lim, img_size, n_iterations)

    numba_res = julia_set(c, lim, img_size, n_iterations)
    Image.fromarray(numba_res).convert('RGB').save(os.path.join(BASE_OUT_FOLDER, 'numba.png'))
    Image.fromarray(original_res).convert('RGB').save(os.path.join(BASE_OUT_FOLDER, 'original.png'))
    assert np.allclose(original_res, numba_res)
    compare_speed()

    profile_file = os.path.join(BASE_OUT_FOLDER, "julia_set.prof")
    cProfile.run("julia_set(c, lim, img_size, n_iterations)", profile_file)

    profile_file = os.path.join(BASE_OUT_FOLDER, "julia_set_jit.prof")
    cProfile.run("julia_set_jit(c, lim, img_size, n_iterations)", profile_file)