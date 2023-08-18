import datasets
import numpy as np

from attack import utils
from torchvision import transforms


# Preprocessing the datasets and DataLoaders creation.
augmentations = transforms.Compose(
    [
        transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(64) if True else transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip() if False else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
    ]
)

def transform_images(examples):
    images = [augmentations(image.convert("RGB")).numpy() for image in examples["image"]]
    return {"input": images}


def data_prepare(data="celeba", mode="ndarry"):
    # supported mode ndarry, datasets
    if data == "celeba":
        files = utils.get_file_names("~/MIA/MIA-Gen/VAEs/data/celeba64/total")
    elif data == "tinyin":
        files = utils.get_file_names("~/MIA/MIA-Gen/VAEs/data/Tiny-IN")
    full_dataset = datasets.Dataset.from_dict({"image": files}).cast_column("image", datasets.Image())
    if mode == "datasets":
        return full_dataset
    elif mode == "ndarry":
        tensor_path = "/mnt/data0/fuwenjie/MIA/MIA-Gen/data/datasets/celeba/celeba64.npz"
        if not utils.check_files_exist(tensor_path):
            print("Don't find the prepared npz files, start to generate!")
            full_dataset.set_transform(transform_images)
            images = []
            for image in full_dataset:
                images.append(image['input'][None, :, :, :])
            full_dataset = np.concatenate(images, axis=0)
            np.savez(tensor_path, full_dataset)
        else:
            full_dataset = np.load(tensor_path)["arr_0"]
        return full_dataset
