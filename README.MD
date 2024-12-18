# Skoltech High Performance Python Lab 2024
## Student: Zabolotnyi Artem


## Create conda env
```bash
conda env create --name hppl -f environment.yml
conda activate hppl
```

### HW1 - Bifurcation map
```bash
python hw1/main.py
```
[Results](hw1/)

### HW2 - Julia set
```bash
python hw2/main.py
```
[Results](hw2/)

### HW3 - Schelling model
```bash
python hw3/main.py
```
[Results](hw3/)

### HW4 - Spectrogram
```bash
python hw4/main.py
```
[Results](hw4/)

### HW5 - Parallel bifurcation map and spectrogram
```bash
python hw5/main.py
```
[Results](hw5/)

### HW6 - Integral
```bash
python hw6/main.py
```
[Results](hw6/)

### HW7 - Column wise shift
```bash
python hw7/main.py
```
[Results](hw7/)

### HW8 - Jitted Mandelbrot set
```bash
python hw8/main.py
```
[Results](hw8/)

### HW9 - Saxpy + CuPy Bifurcation
```bash
conda env create --name hppl_gpu -f environment_gpuenvironment_gpu.yml
conda activate hppl_gpu
```

```bash
python hw9/main.py
```
[Results](hw9/)