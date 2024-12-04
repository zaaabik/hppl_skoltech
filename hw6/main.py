import os
import subprocess
import timeit

from scipy.integrate import quad
import pandas as pd
import rootutils
import numpy as np
from tqdm.auto import tqdm
from glob import glob
import plotly.graph_objects as go

root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from hw6.integral import intergral, FUNC, A, B, N

DEFAULT_SQUARE_PNG_ARGS = dict(width=1024, height=1024, scale=2)
DEFAULT_PNG_ARGS = dict(width=1024, height=512, scale=2)

BASE_OUT_FOLDER = os.path.join(root_path, 'hw6', 'out')
NUM_PROCESSES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 32]


def collect_results():
    files = glob(os.path.join(BASE_OUT_FOLDER, '*result*.csv'))
    all_dfs = []
    for file in files:
        all_dfs.append(pd.read_csv(file))
    total_result_df = pd.concat(all_dfs, axis='rows').reset_index(drop=True)
    total_result_df.to_csv(os.path.join(BASE_OUT_FOLDER, 'final.csv'), index=None)

    mpi_results = total_result_df.loc[
        total_result_df['method'].isin(['mpi', 'single_process'])
    ]
    mpi_results.loc['num_process'] = mpi_results['num_process'].fillna(0)
    mpi_results = mpi_results.sort_values('num_process')
    seq_program_time = mpi_results.loc[mpi_results['method'] == 'single_process']['time_mean'].values[0]
    seq_program_value = total_result_df.loc[total_result_df['method'] == 'library']['result'].values[0]
    result_diff = mpi_results['result'] - seq_program_value

    diff_figure = go.Figure([
        go.Scatter(
            x=mpi_results['num_process'], y=result_diff,
            name='diff',
            yaxis="y1",
        ),
    ],
        layout_title='Difference between out implementation and function from library',
        layout_yaxis_title='Difference',
        layout_xaxis_title='Num processes'
    )
    diff_figure.write_image(os.path.join(BASE_OUT_FOLDER, f'4_difference_of_result.png'),
                            **DEFAULT_PNG_ARGS)

    fig = go.Figure([
        go.Scatter(
            x=mpi_results['num_process'], y=mpi_results['time_mean'],
            name='Execution time',
            yaxis="y1",
            error_y=dict(type='data',
                         array=mpi_results['time_std'], color='purple',
                         thickness=1.5, width=3),
        ),
    ],
        layout_title='Integral execution time (Multiprocessing)',
        layout_yaxis_title='Time (s)',
        layout_xaxis_title='Num processes'
    )
    fig.add_trace(
        go.Scatter(
            x=mpi_results['num_process'], y=seq_program_time / mpi_results['time_mean'].values,
            name='Speedup coefficient',
            yaxis="y4"
        )
    )

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=mpi_results['num_process'],
            ticktext=['Main process only'] + list(mpi_results['num_process'].values[1:])
        ),
        yaxis4=dict(
            side="right",
            title='Speed up coefficient',
            # anchor="free",
            overlaying="y",
        )
    )
    fig.write_image(os.path.join(BASE_OUT_FOLDER, f'4_speed_up_vs_process.png'),
                    **DEFAULT_PNG_ARGS)


if __name__ == "__main__":
    os.makedirs(BASE_OUT_FOLDER, exist_ok=True)
    single_process_result = intergral(FUNC, A, B, N)
    single_process_time = timeit.repeat(lambda:
                                        intergral(FUNC, A, B, N),
                                        number=1, repeat=4)
    single_process_res_dict = {'method': 'single_process',
                               'time_mean': float(np.mean(single_process_time)),
                               'time_std': float(np.std(single_process_time)),
                               'result': single_process_result}

    library_integral, _ = quad(FUNC, A, B)
    library_time = timeit.repeat(lambda: quad(FUNC, A, B), number=1, repeat=4)
    library_res_dict = {'method': 'library',
                        'time_mean': float(np.mean(library_time)),
                        'time_std': float(np.std(library_time)),
                        'result': library_integral}

    df = pd.DataFrame([
        single_process_res_dict, library_res_dict
    ], index=None)
    df.to_csv(
        os.path.join(BASE_OUT_FOLDER, 'result_main_process.csv')
        , index=None
    )

    for num_process in tqdm(NUM_PROCESSES):
        subprocess.run(["mpirun", "-n", str(num_process), "python", "hw6/mpi_integral.py"])

    collect_results()
