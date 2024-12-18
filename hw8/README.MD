### HW8 - Jitted Mandelbrot set
[Source of plots](out)

Numba vs Original implementation:

<img src="out/numba.png" width="256" height="256">
<img src="out/original.png" width="256" height="256">

Speedup:

<img src="out/JIT_speedup.png">

Profile for non-jit:
<img src="out/julia_set_profiler.png">

Jit profiler:
<img src="out/julia_set_jit_profiler.png">

Jit profiler (there is only compiled function call):
<img src="out/julia_set_jit_profiler.png">


