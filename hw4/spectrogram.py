import numpy as np
from numpy import floating, ndarray
from numpy.fft import fft, fftfreq


def create_window_functions(t: np.ndarray[floating], window_width: float,
                            num_cycles: int, function_period: float, num_window_positions: int) -> tuple[
    list[np.ndarray[floating]], np.ndarray
]:
    window_positions = np.linspace(-num_cycles * function_period, +num_cycles * function_period, num_window_positions)
    window_functions = []
    for window_position in window_positions:
        window_functions.append(
            np.exp(-(t - window_position) ** 2 / 2 / window_width ** 2)
        )
    return window_functions, window_positions


def apply_window_function_to_signal(signal: np.ndarray[floating],
                                    window_functions: list[np.ndarray[floating]]) -> list[np.ndarray[floating]]:
    windowed_signals = []
    for window_function in window_functions:
        windowed_signal = signal * window_function
        windowed_signals.append(windowed_signal)
    return windowed_signals


def spectrogram(signal: np.ndarray[floating],
                t: np.ndarray[floating], window_width: float,
                num_cycles: int, function_period: float, num_window_positions: int
                ) -> tuple[list[ndarray[floating]], np.ndarray[floating], np.ndarray[floating]]:
    window_functions, window_positions = create_window_functions(t=t, window_width=window_width, num_cycles=num_cycles,
                                                                 function_period=function_period,
                                                                 num_window_positions=num_window_positions)
    windowed_signals = apply_window_function_to_signal(signal, window_functions)
    sampling_rate = function_period / (t[1] - t[0])
    spectrum = []
    for windowed_signal in windowed_signals:
        fft_values = fft(windowed_signal)
        freqs = fftfreq(len(signal), d=1 / sampling_rate)
        half_of_spectrum = len(freqs) // 2
        freqs = freqs[:half_of_spectrum]
        spectrum.append(np.abs(fft_values[:half_of_spectrum]))
    return np.vstack(spectrum).T, window_positions / function_period, freqs


def stft(signal, window_size, overlap, sampling_rate):
    step = window_size - overlap
    windows = []
    times = []
    freqs = np.fft.fftfreq(window_size, 1 / sampling_rate)

    for start in range(0, len(signal) - window_size, step):
        segment = signal[start:start + window_size]
        windowed_segment = segment * np.hamming(window_size)
        spectrum = np.abs(fft(windowed_segment))[:window_size // 2]
        windows.append(spectrum)
        times.append(start / sampling_rate)

    return np.array(windows).T, freqs[:window_size // 2], times
