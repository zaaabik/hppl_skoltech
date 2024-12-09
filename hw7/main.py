import os
import subprocess

import rootutils
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm

root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


DEFAULT_SQUARE_PNG_ARGS = dict(width=1024, height=1024, scale=2)
DEFAULT_PNG_ARGS = dict(width=1024, height=512, scale=2)

BASE_OUT_FOLDER = os.path.join(root_path, 'hw7', 'out')


def create_shift_animation(image_array, iterations=50):
    frames_list = []
    current_image = image_array.copy()

    for _ in tqdm(range(iterations)):
        buffer = current_image[:, -1]
        current_image[:, 1:] = current_image[:, :-1]
        current_image[:, 0] = buffer
        frames_list.append(current_image.copy())

    imageio.mimsave(os.path.join(BASE_OUT_FOLDER, 'non-parallel.gif'), frames_list, duration=0.1)


if __name__ == '__main__':
    os.makedirs(BASE_OUT_FOLDER, exist_ok=True)

    image_path = os.path.join(root_path, 'hw7', "img.png")
    image = Image.open(image_path)
    image_array = np.array(image)
    create_shift_animation(image_array, iterations=50)
    NUM_PROCESSES = [2, 8, 16]
    for num_process in tqdm(NUM_PROCESSES):
        subprocess.run(["mpirun", "-n", str(num_process), "python", "hw7/mpi_image_shift.py"])
