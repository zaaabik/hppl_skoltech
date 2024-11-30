### HW5 - Parallel bifurcation map and spectrogram
```bash
python hw5/main.py
```
[Source of plots](out)

**Threads vs. Processes**

For the bifurcation map, I use multiprocessing because most of the code is written in pure Python. Due to the Global Interpreter Lock (GIL), multithreading does not improve the execution speed. For the spectrogram, I use multithreading because most of the computational functions rely on NumPy. Since NumPy bypasses the GIL, we can achieve a speedup by using threads instead of processes.

My PC has 6+2 cores:

<img src="pc_spec.png">

Bifurcation Speedup:

<img src="out/1_bifurcation_speed_up.png" width=512 height=512> <img src="out/2_spectrum_speed_up.png" width=512 height=512>

Bifurcation for Multiprocessing and Single Process:

<img src="out/1_bifurcation_plot_num_workers=0.png" width=512 height=512> <img src="out/1_bifurcation_plot_num_workers=8.png" width=512 height=512>

Spectrogram for Multiprocessing and Single Process:

<img src="out/2_spectrogram_num_threads=0.png" width=512 height=512> <img src="out/2_spectrogram_num_threads=8.png" width=512 height=512>