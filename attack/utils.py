import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from torchvision import transforms

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


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


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


def convert_labels_to_one_hot(labels, num_classes):
    '''
    Converts labels of samples from format (N,) to (N, C), where C is the number of classes

    Args:
    labels : numpy array of shape (N,) containing the labels of each sample
    num_classes : integer indicating the total number of classes in the dataset

    Returns:
    numpy array of shape (N, C), where C is the number of classes, containing the one-hot encoded labels
    '''
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    one_hot_labels[np.arange(labels.shape[0]), labels] = 1
    return one_hot_labels


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

def get_file_names(folder_path):
    # List to store the file names
    file_names = []

    # Loop through each file in the folder
    for file_name in sorted(os.listdir(folder_path)):
        # Check if the current item is a file
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(os.path.join(folder_path, file_name))

    return file_names


def transform_images(examples, norm=True):
    # Preprocessing the datasets and DataLoaders creation.
    # if norm: -1 : 1
    augmentations = transforms.Compose(
        [
            transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(64) if True else transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip() if False else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) if norm else transforms.Lambda(lambda x: x),
        ]
    )
    images = [augmentations(image.convert("RGB")) for image in examples["image"]]
    return {"input": images}


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))