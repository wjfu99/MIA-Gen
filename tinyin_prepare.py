import os
import numpy as np
import torch
import pythae
import torch.nn.functional as F
import logging
import random
import sys
from attack.attack_model_diffusion import AttackModel
from pythae.models import AutoModel
from diffusers import DiffusionPipeline
from datasets import Image, Dataset
from collections import OrderedDict
from attack import utils
from tqdm import tqdm
import json
import yaml
from torchvision.transforms import ToPILImage
import datasets
import matplotlib.pyplot as plt

tiny_imagenet_train = datasets.load_dataset('Maysee/tiny-imagenet', split='train')
tiny_imagenet_eval = datasets.load_dataset('Maysee/tiny-imagenet', split='valid')

train_dict = tiny_imagenet_train[:100000]
eval_dict = tiny_imagenet_eval[:10000]

all_dict = {'image': train_dict['image'] + eval_dict['image']}

all_dataset = Dataset.from_dict(all_dict)
img_path = "/mnt/data0/fuwenjie/MIA/MIA-Gen/target_model/data/Tiny-IN/"
for i in tqdm(range(len(all_dataset))):
    all_dataset[i]['image'].save(img_path + f"{i:06}.png", "PNG")

files = utils.get_file_names("/mnt/data0/fuwenjie/MIA/MIA-Gen/target_model/data/Tiny-IN")
total_dataset = datasets.Dataset.from_dict({"image": files}).cast_column("image", Image())