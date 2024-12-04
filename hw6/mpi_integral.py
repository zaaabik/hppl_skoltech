import os
import timeit

import pandas as pd
import rootutils
from mpi4py import MPI
import numpy as np

root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from hw6.integral import FUNC, A, B, N

BASE_OUT_FOLDER = os.path.join(root_path, 'hw6', 'out')


def integrate_parallel(func, a, b, n):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dx = (b - a) / n

    local_n = n // size
    rank_a = a + rank * local_n * dx

    rank_result = 0.0
    for i in range(local_n):
        x = rank_a + i * dx
        rank_result += func(x) * dx

    total_result = comm.reduce(rank_result, op=MPI.SUM, root=0)
    return total_result


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    multi_process_result = integrate_parallel(FUNC, A, B, N)
    multi_process_time = timeit.repeat(lambda:
                                       integrate_parallel(FUNC, A, B, N),
                                       number=1, repeat=4)

    if rank == 0:
        multi_process_res_dict = {'method': 'mpi',
                                  'time_mean': float(np.mean(multi_process_time)),
                                  'time_std': float(np.std(multi_process_time)),
                                  'num_process': size,
                                  'result': multi_process_result}

        df = pd.DataFrame([
            multi_process_res_dict
        ], index=None)
        df.to_csv(os.path.join(BASE_OUT_FOLDER, f'mpi_result_process_{size}.csv'), index=None)
