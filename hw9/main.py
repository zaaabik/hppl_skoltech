import os
import timeit

import numpy as np

import rootutils
import cupy as cp
from plotly import graph_objs as go



root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from hw1.bifurcation_map import get_bifurcation_data
from hw5.mp_alogrithms import bifurcation_mp

BASE_OUT_FOLDER = os.path.join(root_path, 'hw9', 'out')

DEFAULT_SQUARE_PNG_ARGS = dict(width=1024, height=1024, scale=1)
DEFAULT_PNG_ARGS = dict(width=1024, height=512, scale=1)


def logistic_map_cp(x, r):
    return r * x * (1 - x)


def get_bifurcation_data_cp(x0, r_values, n, m):
    total_x = []
    total_r = []
    res = cp.full_like(r_values, x0)

    for i in range(n):
        res = logistic_map_cp(res, r_values)

    for i in range(m):
        res = logistic_map_cp(res, r_values)
        total_x.append(res.copy())
        total_r.append(r_values.copy())
    return cp.concatenate(total_x), cp.concatenate(total_r)


def cupy_saxpy(a, x, y):
    res = a * x * y
    cp.cuda.Device().synchronize()
    return res


def measure_cupy_saxpy(n, repeat=4):
    x = cp.random.rand(n, dtype=cp.float32)
    y = cp.random.rand(n, dtype=cp.float32)
    a = 4.

    time = timeit.repeat(
        lambda: cupy_saxpy(a, x, y),
        number=1, repeat=repeat
    )

    return np.mean(time).item(), np.std(time).item()


def measure_numpy_saxpy(n: int = 10 ** 4, repeat=4):
    x = np.random.rand(n).astype('float32')
    y = np.random.rand(n).astype('float32')
    a = 4.
    time = timeit.repeat(lambda:
                         a * x + y,
                         number=1, repeat=repeat)

    return np.mean(time).item(), np.std(time).item()


def measure_time():
    sizes = [
        10 ** i for i in [1, 3, 4, 7, 8, 9]
    ]
    cupy_times = []
    numpy_times = []

    for size in sizes:
        cupy_times.append(measure_cupy_saxpy(size))

    for size in sizes:
        numpy_times.append(measure_numpy_saxpy(size))

    cupy_scatter = go.Scatter(x=sizes, y=[i[0] for i in cupy_times], name='CuPy',
                              error_y=dict(type='data', array=[i[1] for i in cupy_times])
                              )

    numpy_scatter = go.Scatter(x=sizes, y=[i[0] for i in numpy_times], name='NumPy',
                               error_y=dict(type='data', array=[i[1] for i in numpy_times])
                               )

    fig = go.Figure([cupy_scatter, numpy_scatter],
                    layout_xaxis_type='log', layout_yaxis_type='log',
                    layout_xaxis_title='Vector size log scale',
                    layout_title='Compare NumPy and CuPy saxpy',
                    layout_yaxis_title='Computation time log scale (s)')

    fig.write_image(os.path.join(BASE_OUT_FOLDER, 'saxpy.png'), **DEFAULT_PNG_ARGS)


R_NUMBERS = [4000, 6000, 8000, 12000]
n = 2048
m = 128
x0 = 0.5


def measure_bifurcation_base(repeat=4):
    times = []
    stds = []
    for r in R_NUMBERS:
        r_space = np.linspace(2.5, 4, r).tolist()
        time = timeit.repeat(lambda:
                             get_bifurcation_data(x0, r_space, n, m),
                             number=1, repeat=repeat)

        times.append(np.mean(time))
        stds.append(np.std(time))
    return times, stds


def measure_bifurcation_mp(num_workers, repeat=4):
    times = []
    stds = []

    for r in R_NUMBERS:
        r_space = np.linspace(2.5, 4, r).tolist()
        time = timeit.repeat(lambda:
                             bifurcation_mp(x0, r_space, n, m, num_workers),
                             number=1, repeat=repeat)

        times.append(np.mean(time))
        stds.append(np.std(time))
    return times, stds


def call_and_sync(func):
    func()
    cp.cuda.Device().synchronize()


def measure_bifurcation_cupy(repeat=4):
    times = []
    stds = []

    for r in R_NUMBERS:
        r_space = cp.linspace(2.5, 4, r)
        time = timeit.repeat(lambda:
                             call_and_sync(lambda: get_bifurcation_data_cp(x0, r_space, n, m)),
                             number=1, repeat=repeat)

        times.append(np.mean(time))
        stds.append(np.std(time))
    return times, stds


def measure_bifurcation():
    cupy_time, cupy_std = measure_bifurcation_cupy()
    mp_scatters = []
    for num_worker in [1, 2, 4, 8]:
        mp_time, mp_std = measure_bifurcation_mp(num_worker)

        mp_scatter = go.Scatter(
            x=R_NUMBERS, y=mp_time, name=f'MP_num_processes={num_worker}',
            opacity=(num_worker / 8), marker_color='black',
            error_y=dict(type='data', array=mp_std)
        )
        mp_scatters.append(mp_scatter)
    base_time, base_std = measure_bifurcation_base()

    base_scatter = go.Scatter(
        x=R_NUMBERS, y=base_time, name='Base',
        error_y=dict(type='data', array=base_std)
    )

    cupy_scatter = go.Scatter(
        x=R_NUMBERS, y=cupy_time, name='CUPY',
        error_y=dict(type='data', array=cupy_std)
    )

    fig = go.Figure([
        base_scatter,
        cupy_scatter,
        *mp_scatters
        ],
        layout_xaxis_title='R count', layout_yaxis_type='log',
        layout_title='Compare NumPy, CuPy, Multiprocess bifurcation',
        layout_yaxis_title='Computation time log scale (s)')

    fig.write_image(os.path.join(BASE_OUT_FOLDER, 'bifrucation_compare.png'), **DEFAULT_PNG_ARGS)


if __name__ == '__main__':
    os.makedirs(BASE_OUT_FOLDER, exist_ok=True)
    measure_time()
    measure_bifurcation()



