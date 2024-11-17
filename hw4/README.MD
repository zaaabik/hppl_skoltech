### HW3 - Schelling model
[Source of plots](out)

Base signal:

<img src="out/signal.png" height=512 width=768>

Base fft:

<img src="out/fft.png" height=512 width=768>


Base signal + frequency = 4 and time_shift = 7 cycles:

<img src="out/signal.png" height=512 width=768>

Base signal + frequency = 4 and time_shift = 7 cycles fft:

<img src="out/fft_with_4th_wave_packet.png" height=512 width=768>

Window function for different position:

<img src="out/windowed_function.png" height=512 width=768>

Window function applied to signal:

<img src="out/windowed_signal.png" height=512 width=768>

Spectrogram:

<img src="out/spectrogram.png" height=512 width=768>

FFT time measure:

<img src="out/fft_time.png" heigh=512 width=768>

**Question:** Measure the timing, can you explain the difference between speed of FFT for different sample size? Write something as a possible explanation. (2 points)

**Answer:**
The Fast Fourier Transform (FFT) can achiлeve the best 
performance using a divide-and-conquer approach when you 
split the sequence into two parts and apply the algorithm to each part. 
If your sequence has a length that is a power of 2, you can efficiently split this sequence into two parts until the sequence length is reduced to 1