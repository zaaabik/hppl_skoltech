import os.path
import timeit

import numpy as np
import plotly.graph_objects as go
import rootutils
from numpy import floating
from numpy.fft import fftfreq, fft
from math import pi

root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from hw4.spectrogram import create_window_functions, spectrogram, apply_window_function_to_signal

BASE_OUT_FOLDER = os.path.join(root_path, 'hw4', 'out')
DEFAULT_PNG_ARGS = dict(width=1024, height=768, scale=4)
TIME_DOMAIN_AXIS = dict(layout_xaxis_title='Cycles', layout_yaxis_title='Amplitude')

function_period = 2 * pi


def create_signal(num_samples=2 ** 14):
    t = np.linspace(-20 * function_period, 20 * function_period, num_samples)

    y = np.sin(t) * np.exp(-t ** 2 / 2 / 20 ** 2)
    y = y + np.sin(3 * t) * np.exp(-(t - 5 * function_period) ** 2 / 2 / 20 ** 2)
    y = y + np.sin(5.5 * t) * np.exp(-(t - 10 * function_period) ** 2 / 2 / 5 ** 2)
    return y, t


def add_wave_packet(y: np.ndarray, frequency: float, time_shift: float) -> np.ndarray:
    y = y.copy()
    y = y + np.sin(frequency * t) * np.exp(-(t - time_shift * function_period) ** 2 / 2 / 20 ** 2)
    return y


def add_wave_packet_fft(signal: np.ndarray[floating], t: np.ndarray[floating]):
    sampling_rate = function_period / (t[1] - t[0])
    fig = go.Figure(go.Scatter(
        x=t / function_period, y=signal
    ), **TIME_DOMAIN_AXIS, layout_title='Signal', layout_yaxis_range=[-2, 2])
    fig.write_image(
        os.path.join(BASE_OUT_FOLDER, 'signal.png'),
        **DEFAULT_PNG_ARGS
    )

    fft_values = fft(signal)
    freqs = fftfreq(len(signal), d=1 / sampling_rate)
    half_of_spectrum = len(freqs) // 2
    fig_fft = go.Figure(
        data=go.Scatter(x=freqs[:half_of_spectrum], y=np.abs(fft_values[:half_of_spectrum]), mode='lines'),
        layout_title="FFT Spectrum",
        layout_xaxis_title="Frequency",
        layout_yaxis_title="Amplitude",
        layout_xaxis_range=[0, 10]
    )
    fig_fft.write_image(os.path.join(BASE_OUT_FOLDER, 'fft.png'), **DEFAULT_PNG_ARGS)

    signal = add_wave_packet(signal, 4, 7)

    fig = go.Figure(go.Scatter(
        x=t / function_period, y=signal
    ), **TIME_DOMAIN_AXIS, layout_title='Signal with 4th wave packet', layout_yaxis_range=[-2, 2])
    fig.write_image(os.path.join(BASE_OUT_FOLDER, 'signal_with_wave_packet.png'), **DEFAULT_PNG_ARGS)

    fft_values = fft(signal)
    freqs = fftfreq(len(signal), d=1 / sampling_rate)
    half_of_spectrum = len(freqs) // 2
    fig_fft = go.Figure(
        data=go.Scatter(x=freqs[:half_of_spectrum], y=np.abs(fft_values[:half_of_spectrum]), mode='lines'),
        layout_title="FFT Spectrum",
        layout_xaxis_title="Frequency",
        layout_yaxis_title="Amplitude",
        layout_xaxis_range=[0, 10]
    )
    fig_fft.write_image(os.path.join(BASE_OUT_FOLDER, 'fft_with_4th_wave_packet.png'), **DEFAULT_PNG_ARGS)
    return signal


def plot_spectrogram(signal: np.ndarray[floating], t: np.ndarray[floating]):
    num_cycles = 20
    num_window_positions = 20
    window_functions, window_positions = create_window_functions(t, window_width=function_period,
                                                                 num_cycles=num_cycles, function_period=function_period,
                                                                 num_window_positions=num_window_positions)
    window_function_scatters = []
    for window_function, window_position in zip(window_functions, window_positions):
        window_function_scatters.append(
            go.Scatter(x=t / function_period, y=window_function,
                       name=f'Window position={window_position / function_period:.3f} cycles')
        )
    window_function_fig = go.Figure(window_function_scatters,
                                    layout_title='Window filters',
                                    **TIME_DOMAIN_AXIS)
    window_function_fig.write_image(os.path.join(BASE_OUT_FOLDER, 'windowed_function.png'), **DEFAULT_PNG_ARGS)

    windowed_signals = apply_window_function_to_signal(signal, window_functions)
    scatters = []
    for windowed_signal, window_position in zip(windowed_signals, window_positions):
        scatters.append(go.Scatter(x=t / function_period, y=windowed_signal,
                                   name=f'Window position cycles {window_position / function_period:.3f}'))
    scatters.append(go.Scatter(x=t / function_period, y=signal, name='Original signal', opacity=0.4))
    fig = go.Figure(scatters, layout_title='Windowed signal', **TIME_DOMAIN_AXIS)
    fig.write_image(os.path.join(BASE_OUT_FOLDER, 'windowed_signal.png'), **DEFAULT_PNG_ARGS)

    spectrum, window_positions, freqs = spectrogram(signal, t, window_width=function_period,
                                                    num_cycles=num_cycles, function_period=function_period,
                                                    num_window_positions=num_window_positions)
    fig_spec = go.Figure(
        data=go.Heatmap(
            z=spectrum, x=window_positions, y=freqs, colorscale='Viridis'
        ),
        layout_title="Spectrogram", layout_xaxis_title="Cycles", layout_yaxis_title="Frequency",
        layout_yaxis_range=[0, 10]
    )
    fig_spec.write_image(
        os.path.join(
            BASE_OUT_FOLDER, 'spectrogram.png'
        ),
        **DEFAULT_PNG_ARGS
    )


def measure_fft_time():
    nums_samples = [2 ** 14 - 125, 2 ** 14 - 5, 2 ** 14, 2 ** 14 + 5]
    means = []
    stds = []
    for samples in nums_samples:
        y = create_signal(samples)

        def get_fft():
            fft(y)

        func_time = timeit.repeat(get_fft, number=10, repeat=10)

        means.append(np.mean(func_time).tolist())
        stds.append(np.std(func_time).tolist())
    fig = go.Figure(go.Scatter(x=nums_samples, y=means,
                               error_y=dict(type='data',
                                            array=stds, color='purple',
                                            thickness=1.5, width=3)
                               ),
                    layout_title='FFT speed', layout_yaxis_title='Time (s)', layout_xaxis_title='Num samples',
                    layout_yaxis_type='log'
                    )
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=nums_samples,
            ticktext=['2 ** 14 - 125', '2 ** 14 - 5', '2 ** 14', '2 ** 14 + 5']
        ))
    fig.write_image(
        os.path.join(
            BASE_OUT_FOLDER, 'fft_time.png'
        ),
        **DEFAULT_PNG_ARGS
    )


if __name__ == '__main__':
    os.makedirs(BASE_OUT_FOLDER, exist_ok=True)
    signal, t = create_signal()
    # 1. Add 4th wave packet (frequency = 4 and time_shift = 7 cycles).
    # Demonstrate the effect on the plot of the FFT spectrum (1 point)
    signal = add_wave_packet_fft(signal, t)

    # 2. Implement the spectrogram, show the effect of (1) on the spectrogram. Don’t forget to label the axes (2 points)
    plot_spectrogram(signal, t)

    # 3. Change the number of time steps in your signal to the power of 2 (i.e. 2**14) and then slightly change
    # the number of timesteps (i.e 2**14 +- 5).
    # Measure the timing, can you explain the difference? Write something as a possible explanation. (2 points)
    measure_fft_time()

