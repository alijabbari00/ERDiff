import numpy as np
import os
import pandas as pd
import torch


def load_tensors_by_index(i: int, dataset_folder: str) -> (torch.Tensor, torch.Tensor):
    """
    Given an integer index i, loads the ith file in `file_list`
    (appending '.npz'), and returns x, y as torch.Tensors.
    """

    file_list = os.listdir(dataset_folder)

    if not (0 <= i < len(file_list)):
        raise IndexError(f"Index {i} out of range [0, {len(file_list)})")

    fname = file_list[i]
    path = os.path.join(dataset_folder, fname)

    # load numpy arrays
    with np.load(path) as data:
        x_np = data['x']
        y_np = data['y']

    x_np = torch.from_numpy(x_np)
    y_np = torch.from_numpy(y_np)

    return x_np, y_np


def get_dataset_num_days(dataset_folder: str):
    return len(os.listdir(dataset_folder))


def load_dummy_dataset(num_samples: int = 100):
    """Return dummy tensors mimicking the new dataset.

    Parameters
    ----------
    num_samples : int, optional
        Number of samples to generate.

    Returns
    -------
    x : torch.Tensor
        Dummy input tensor of shape ``(num_samples, 95)``.
    y : torch.Tensor
        Dummy target tensor of shape ``(num_samples, 6)``.
    """

    x = torch.randn(num_samples, 95)
    y = torch.randn(num_samples, 6)
    return x, y
