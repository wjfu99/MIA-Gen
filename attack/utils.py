import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

class Dict(dict):
    def __getattr__(self, name):
        if name in self:
            return  self[name]
        raise AttributeError(f"'Dict' object has no attribute '{name}'")
    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

def check_files_exist(*file_paths):
    """
    Check if the input file(s) exist at the given file path(s).

    Parameters:
        *file_paths (str): One or more strings representing the file path(s) to check.

    Returns:
        bool: True if all the files exist, False otherwise.
    """
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            return False
    return True


def save_dict_to_npz(my_dict, file_path):
    """
    Saves a dictionary with ndarray values to an npz file.

    Parameters:
        my_dict (dict): A dictionary with ndarray values to be saved.
        file_path (str): The file path to save the dictionary values to.

    Returns:
        None
    """
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(file_path, 'wb') as f:
        np.savez(f, **my_dict)


def load_dict_from_npz(file_path):
    """
    Loads a dictionary with ndarray values from an npz file.

    Parameters:
        file_path (str): The file path of the npz file to load.

    Returns:
        dict: A dictionary containing the values stored in the npz file.
    """
    with np.load(file_path) as data:
        my_dict = Dict({key: value for key, value in data.items() if isinstance(value, np.ndarray)})
    return my_dict


def ndarray_to_tensor(*ndarrays):
    """
    Converts multiple numpy ndarrays to PyTorch tensors.

    Parameters:
        *ndarrays (numpy.ndarray): Multiple numpy ndarrays to convert.

    Returns:
        tuple of torch.Tensor: A tuple of PyTorch tensors with the same data as the input ndarrays.
    """
    tensors = tuple(torch.from_numpy(ndarray).cuda().float() for ndarray in ndarrays)
    return tensors


def tensor_to_ndarray(*tensors):
    """
    Converts multiple PyTorch tensors to numpy ndarrays.

    Parameters:
        *tensors (torch.Tensor): Multiple PyTorch tensors to convert.

    Returns:
        tuple of numpy.ndarray: A tuple of numpy ndarrays with the same data as the input tensors.
    """
    ndarrays = tuple(tensor.detach().cpu().numpy() for tensor in tensors)
    return ndarrays


def show_image(image):
    # show reconstructions
    if image.shape[0] == 1:
        fig, axes = plt.subplots(figsize=(10, 10))
        axes.imshow(image.cpu().squeeze(), cmap='gray')
        axes.axis('off')
        plt.tight_layout(pad=0.)
        plt.show()
    elif image.shape[0] == 3:
        pil_image = ToPILImage()(image.cpu())
        fig, axes = plt.subplots(figsize=(10, 10))
        axes.imshow(pil_image)
        axes.axis('off')
        plt.tight_layout(pad=0.)
        plt.show()