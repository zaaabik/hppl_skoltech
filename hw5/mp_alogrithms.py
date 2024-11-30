from functools import partial
from multiprocessing.pool import ThreadPool, Pool
from typing import Any

import numpy as np
from numpy import floating, ndarray, dtype
from numpy.fft import fft, fftfreq

from hw1.bifurcation_map import get_bifurcation_data


def get_bifurcation_data_mp(*args, **kwargs):
    total_x, total_r = get_bifurcation_data(*args, **kwargs)
    return np.array(total_x), np.array(total_r)  # to improve communication between processes


def bifurcation_mp(x0, r, n, m, num_workers):
    p_func = partial(get_bifurcation_data_mp, x0, n=n, m=m)
    with Pool(num_workers) as pool:
        out = pool.map(
            p_func, np.array_split(r, num_workers)
        )
    total_x = []
    total_r = []
    for x, r in out:
        total_x.append(x)
        total_r.append(r)

    x = np.concat(total_x)
    r = np.concat(total_r)
    return x, r


def spectrogram_mp(signal: np.ndarray[floating],
                   t: np.ndarray[floating], window_width: float,
                   num_cycles: int, function_period: float, num_window_positions: int, num_workers: int = 1
                   ) -> tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[floating[Any]]]]:

    window_positions = np.linspace(-num_cycles * function_period, +num_cycles * function_period, num_window_positions)
    with ThreadPool(num_workers) as pool:
        func = partial(spectrogram_part,
                       signal, t, window_width,num_cycles,
                       function_period, num_window_positions)
        res = pool.map(func, range(num_window_positions))

    sampling_rate = function_period / (t[1] - t[0])
    freqs = fftfreq(len(signal), d=1 / sampling_rate)
    half_of_spectrum = len(freqs) // 2

    spectrum = np.vstack(res)[:half_of_spectrum].T
    freqs = freqs[:half_of_spectrum]
    return spectrum, window_positions / function_period, freqs


def create_window_function(t: np.ndarray[floating], window_width: float,
                           num_cycles: int, function_period: float, num_window_positions: int,
                           current_window_position: int
                           ) -> np.ndarray:
    window_position = np.linspace(-num_cycles * function_period, +num_cycles * function_period, num_window_positions)[
        current_window_position
    ]
    return np.exp(-(t - window_position) ** 2 / 2 / window_width ** 2)


def spectrogram_part(signal: np.ndarray[floating],
                     t: np.ndarray[floating], window_width: float,
                     num_cycles: int, function_period: float, num_window_positions: int, current_window_position: int):
    window_function = create_window_function(t=t, window_width=window_width, num_cycles=num_cycles,
                                             function_period=function_period, num_window_positions=num_window_positions,
                                             current_window_position=current_window_position)
    windowed_signal = signal * window_function
    fft_values = fft(windowed_signal)
    return np.abs(fft_values)
