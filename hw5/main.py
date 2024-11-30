import timeit

import numpy as np
import rootutils
import plotly.graph_objects as go
import os
from numpy import floating

root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from hw1.bifurcation_map import get_bifurcation_data
from hw4.main import function_period, create_signal
from hw4.spectrogram import spectrogram
from hw5.mp_alogrithms import spectrogram_mp, bifurcation_mp

BASE_OUT_FOLDER = os.path.join(root_path, 'hw5', 'out')
DEFAULT_PNG_ARGS = dict(width=1024, height=512, scale=2)
DEFAULT_SQUARE_PNG_ARGS = dict(width=1024, height=1024, scale=2)

NUM_WORKERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32]
R_NUMBERS = 6000
n = 2048
m = 128


def create_bifurcation_plot_mp(r_numbers: int = R_NUMBERS, num_workers: int = 4, save_output: bool = False):
    r = np.linspace(2.5, 4, r_numbers).tolist()
    x0 = 0.5
    if num_workers == 0:
        x, r = get_bifurcation_data(x0, r, n, m)
    else:
        x, r = bifurcation_mp(x0=x0, r=r, n=n, m=m, num_workers=num_workers)

    if save_output:
        fig = go.Figure(go.Scatter(x=r, y=x, mode='markers', marker={'size': 2.5}))
        fig.update_layout(title=f'Bifurcation map num workers = {num_workers}', xaxis={'title': 'r'},
                          yaxis={'title': 'x'})

        fig.write_image(os.path.join(BASE_OUT_FOLDER, f'1_bifurcation_plot_num_workers={num_workers}.png'),
                        **DEFAULT_SQUARE_PNG_ARGS)


def speedup_for_bifurcation():
    mean_time = []
    std_time = []

    for worker_number in NUM_WORKERS:
        time = timeit.repeat(lambda:
                             create_bifurcation_plot_mp(num_workers=worker_number),
                             number=1, repeat=2)
        mean_time.append(np.mean(time))
        std_time.append(np.std(time))

    fig = go.Figure(
        go.Scatter(
            x=NUM_WORKERS, y=mean_time,
            error_y=dict(type='data',
                         array=std_time, color='purple',
                         thickness=1.5, width=3)
        ),
        layout_title='Bifurcation speed up (Multithreading)',
        layout_yaxis_title='Time (s)',
        layout_xaxis_title='Num processes'
    )

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=NUM_WORKERS,
            ticktext=['Main process only'] + NUM_WORKERS[1:]
        ))

    fig.write_image(os.path.join(BASE_OUT_FOLDER, f'1_bifurcation_speed_up.png'), **DEFAULT_SQUARE_PNG_ARGS)


def plot_spectrogram(signal: np.ndarray[floating], t: np.ndarray[floating],
                     num_workers: int = 0, save_output=False,
                     num_window_positions: int = 4096, num_cycles: int = 20):
    if num_workers == 0:
        spectrum, window_positions, freqs = spectrogram(signal, t, window_width=function_period,
                                                        num_cycles=num_cycles, function_period=function_period,
                                                        num_window_positions=num_window_positions)
    else:
        spectrum, window_positions, freqs = spectrogram_mp(signal, t, window_width=function_period,
                                                           num_cycles=num_cycles, function_period=function_period,
                                                           num_window_positions=num_window_positions,
                                                           num_workers=num_workers)
    if save_output:
        fig_spec = go.Figure(
            data=go.Heatmap(
                z=spectrum, x=window_positions, y=freqs, colorscale='Viridis'
            ),
            layout_title=f"Spectrogram num threads = {num_workers}", layout_xaxis_title="Cycles",
            layout_yaxis_title="Frequency",
            layout_yaxis_range=[0, 10]
        )
        fig_spec.write_image(
            os.path.join(
                BASE_OUT_FOLDER, f'2_spectrogram_num_threads={num_workers}.png'
            ),
            **DEFAULT_PNG_ARGS
        )


def speedup_for_spectrum():
    mean_time = []
    std_time = []

    signal, t = create_signal(2 ** 15)

    for worker_number in NUM_WORKERS:
        print(worker_number)
        time = timeit.repeat(lambda:
                             plot_spectrogram(signal, t, num_workers=worker_number),
                             number=1, repeat=2)
        mean_time.append(np.mean(time))
        std_time.append(np.std(time))

    fig = go.Figure(go.Scatter(x=NUM_WORKERS, y=mean_time,
                               error_y=dict(type='data',
                                            array=std_time, color='purple',
                                            thickness=1.5, width=3)
                               ),
                    layout_title='Spectrum speed up (Multithreading)', layout_yaxis_title='Time (s)',
                    layout_xaxis_title='Num threads'
                    )
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=NUM_WORKERS,
            ticktext=['Main process only'] + NUM_WORKERS[1:]
        ))

    fig.write_image(os.path.join(BASE_OUT_FOLDER, f'2_spectrum_speed_up.png'),
                    **DEFAULT_SQUARE_PNG_ARGS)


if __name__ == "__main__":
    os.makedirs(BASE_OUT_FOLDER, exist_ok=True)

    # 1. Implement parallel version of bifurcation map (2 points)
    create_bifurcation_plot_mp(r_numbers=256, num_workers=0, save_output=True)
    create_bifurcation_plot_mp(r_numbers=256, num_workers=8, save_output=True)

    # 2. Plot speedup vs. number of processes/threads (2 points)
    speedup_for_bifurcation()

    # 3. Implement parallel version of spectrogram (2 points)
    signal, t = create_signal(2 ** 14)
    plot_spectrogram(signal, t, num_workers=0, num_window_positions=128, save_output=True)
    plot_spectrogram(signal, t, num_workers=8, num_window_positions=128, save_output=True)

    # 4. Plot speedup vs. number of processes/threads (2 points)
    speedup_for_spectrum()
