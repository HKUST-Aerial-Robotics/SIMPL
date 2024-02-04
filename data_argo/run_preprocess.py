"""This script is for dataset preprocessing."""

import os
from os.path import expanduser
import time
from typing import Any, Dict, List, Tuple
import random
#
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pandas as pd
import pickle as pkl
#
from argo_preprocess import ArgoPreproc

_FEATURES_SMALL_SIZE = 1024


def parse_arguments() -> Any:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="",
        type=str,
        help="Directory where the sequences (csv files) are saved",
    )
    parser.add_argument(
        "--save_dir",
        default="./dataset_argo/features/",
        type=str,
        help="Directory where the computed features are to be saved",
    )
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        help="train/val/test",
    )
    parser.add_argument(
        "--obs_len",
        default=20,
        type=int,
        help="Observed length of the trajectory",
    )
    parser.add_argument(
        "--pred_len",
        default=30,
        type=int,
        help="Prediction Horizon",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="If true, a small subset of data is used.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If true, debug mode.",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="If true, viz.",
    )
    return parser.parse_args()


def load_seq_save_features(args: Any, start_idx: int, batch_size: int, sequences: List[str],
                           save_dir: str, thread_idx: int) -> None:
    """ Load sequences, compute features, and save them """
    # print('thread_idx: ', thread_idx, ' start_idx: ', start_idx)
    dataset = ArgoPreproc(args, verbose=False)

    # Enumerate over the batch starting at start_idx
    for seq in sequences[start_idx:start_idx + batch_size]:
        if not seq.endswith(".csv"):
            continue

        seq_id = int(seq.split(".")[0])
        seq_path = f"{args.data_dir}/{seq}"

        df = pd.read_csv(seq_path, dtype={'TIMESTAMP': str,
                                          'TRACK_ID': str,
                                          'OBJECT_TYPE': str,
                                          'X': float,
                                          'Y': float,
                                          'CITY_NAME': str})
        data, headers = dataset.process(seq_id, df)

        if not args.debug:
            data_df = pd.DataFrame(data, columns=headers)
            filename = '{}'.format(data[0][0])
            data_df.to_pickle(f"{save_dir}/{filename}.pkl")  # compression='gzip'

    print('Finish computing {} - {}'.format(start_idx, start_idx + batch_size))


if __name__ == "__main__":
    """Load sequences and save the computed features."""
    start = time.time()
    args = parse_arguments()

    sequences = os.listdir(args.data_dir)
    # random.shuffle(sequences)

    num_sequences = _FEATURES_SMALL_SIZE if args.small else len(sequences)
    sequences = sequences[:num_sequences]
    print("Num of sequences: ", num_sequences)

    # ! You can directly set the number of CPU cores to use
    n_proc = multiprocessing.cpu_count() - 2 if not args.debug else 1

    batch_size = np.max([int(np.ceil(num_sequences / n_proc)), 1])
    print('n_proc: {}, batch_size: {}'.format(n_proc, batch_size))

    save_dir = args.save_dir + f"{args.mode}"
    os.makedirs(save_dir, exist_ok=True)
    print('save processed dataset to {}'.format(save_dir))

    Parallel(n_jobs=n_proc)(delayed(load_seq_save_features)(args, i, batch_size, sequences, save_dir, k)
                            for i, k in zip(range(0, num_sequences, batch_size), range(len(range(0, num_sequences, batch_size)))))

    print(f"Preprocess for {args.mode} set completed in {(time.time()-start)/60.0:.2f} mins")
