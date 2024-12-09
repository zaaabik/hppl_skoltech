import imageio
from mpi4py import MPI
import numpy as np
from PIL import Image
import rootutils
import os

root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
DEFAULT_SQUARE_PNG_ARGS = dict(width=1024, height=1024, scale=2)
DEFAULT_PNG_ARGS = dict(width=1024, height=512, scale=2)

BASE_OUT_FOLDER = os.path.join(root_path, 'hw7', 'out')


def shift_image(local_image):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    send_to_rank = (rank + 1) % size
    receive_from_rank = (rank - 1) % size

    column_to_send = local_image[-1]
    buffer_from_left = np.empty((1, rows, channels), dtype=image_array.dtype)

    comm.Send(np.array(column_to_send), dest=send_to_rank)
    comm.Recv(buffer_from_left, source=receive_from_rank)
    local_image[1:] = local_image[:-1]
    local_image[0] = buffer_from_left
    return local_image.copy()


def get_columns_per_rank(local_rank, world_size, total_columns):
    columns_per_rank = total_columns // world_size
    reminder = total_columns % size
    if local_rank < reminder:
        columns_per_rank += 1
    return columns_per_rank


if __name__ == '__main__':
    # Load the image and start the MPI process
    image = Image.open(os.path.join(root_path, 'hw7', "img.png"))
    image_array = np.array(image)

    # Only rank 0 should execute the visualization code
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rows, cols, channels = image_array.shape
    image_array = np.ascontiguousarray(image_array.transpose(1, 0, 2))

    columns_per_rank = get_columns_per_rank(rank, size, cols)
    left_local_bound = sum(get_columns_per_rank(r, size, cols) for r in range(rank))
    right_local_bound = left_local_bound + columns_per_rank

    local_image = image_array[left_local_bound:right_local_bound, :, :]
    local_images_during_shift = []
    num_iterations = 50
    for _ in range(num_iterations):
        local_images_during_shift.append(local_image[:, None])
        local_image = shift_image(local_image)
    local_images_during_shift.append(local_image[:, None])

    local_images_during_shift = np.stack(local_images_during_shift, axis=1)

    if rank == 0:
        final_image = np.zeros((cols, (num_iterations + 1), rows, channels), dtype=image_array.dtype)
        recv_counts = [
            (num_iterations + 1) * get_columns_per_rank(i, size, cols) * rows * channels
            for i in range(size)
        ]
        assert len(final_image.flatten()) == sum(recv_counts)
        comm.Gatherv(local_images_during_shift, [final_image, recv_counts, None, MPI.UNSIGNED_CHAR], root=0)
        frames_list = final_image.transpose(1, 2, 0, 3)
        imageio.mimsave(os.path.join(BASE_OUT_FOLDER, f'parallel_world_size={size}.gif'), list(frames_list), duration=0.1)

    else:
        comm.Gatherv(local_images_during_shift, None, root=0)
