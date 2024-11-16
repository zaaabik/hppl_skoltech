import os
import imageio
from imageio.v3 import imread
from plotly.graph_objs import Figure

TMP_FIG_DIR = 'tmp_fig_dir'


def write_gif_from_figures(figures: list[Figure], output_filename: str, fig_sec_duration=0.5):
    os.makedirs(TMP_FIG_DIR, exist_ok=True)
    file_names = []
    for idx, figure in enumerate(figures):
        file_name = os.path.join(TMP_FIG_DIR, f'{idx}.png')
        figure.write_image(file_name)
        file_names.append(file_name)

    with imageio.get_writer(output_filename, mode='I', duration=fig_sec_duration) as writer:
        for file_name in file_names:
            image = imread(file_name)
            writer.append_data(image)

    for frame in file_names:
        os.remove(frame)

    os.rmdir(TMP_FIG_DIR)
