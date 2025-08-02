import os

import numpy as np

import constants


def get_dataset_folder(dataset_name: str):
    data_folder = constants.DATA_PATH
    return os.path.join(data_folder, dataset_name)


def load_tensors_by_index(i: int, dataset_name: str) -> (np.ndarray, np.ndarray):
    """
    Given an integer index i, loads the ith file in `file_list`
    (appending '.npz'), and returns x, y as torch.Tensors.
    """
    if '/' in dataset_name:
        dataset_name = dataset_name.split('/')[-1]
    dataset_folder = get_dataset_folder(dataset_name)

    file_list = os.listdir(dataset_folder)

    if not (0 <= i < len(file_list)):
        raise IndexError(f"Index {i} out of range [0, {len(file_list)})")

    fname = file_list[i]
    path = os.path.join(dataset_folder, fname)

    # load numpy arrays
    with np.load(path) as data:
        x_np = data['x']
        y_np = data['y']

    return x_np, y_np


def cut_trials(x: np.ndarray, y: np.ndarray, trial_length: int) -> (np.ndarray, np.ndarray):
    """
    given x and y of shape [N, a] and [N, b], cut into trials of length `trial_length`
    and return a resulting x and y of shape [N/trial_length, trial_length, a] and [N/trial_length, trial_length, b]
    """
    n_full_trials = x.shape[0] // trial_length
    usable_rows = n_full_trials * trial_length
    x_trimmed = x[:usable_rows]
    y_trimmed = y[:usable_rows]
    pieces_x = np.split(x_trimmed, n_full_trials)
    pieces_y = np.split(y_trimmed, n_full_trials)
    stacked_x = np.stack(pieces_x)
    stacked_y = np.stack(pieces_y)
    return stacked_x, stacked_y


def get_dataset_num_days(dataset_folder: str):
    return len(os.listdir(dataset_folder))


def get_batches(x, batch_size):
    n_batches = len(x) // (batch_size)
    x = x[:n_batches * batch_size:]
    for n in range(0, x.shape[0], batch_size):
        x_batch = x[n:n + (batch_size)]
        yield x_batch
