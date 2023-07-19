import os
import numpy as np
import torch
import pythae
import torch.nn.functional as F
import logging
import random

from attack.attack_model_diffusion import AttackModel
from pythae.models import AutoModel
from diffusers import DiffusionPipeline
from datasets import Image, Dataset
from collections import OrderedDict
from attack import utils
import json
import yaml
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

cfg = {
    "sample_number": 10000
}

img_shape = (3, 64, 64)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


PATH = os.path.dirname(os.path.abspath(__file__))
# Automatically select the freest GPU.
os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_available))
device = "cuda" + ":" + str(np.argmax(memory_available))
torch.cuda.set_device(device)

target_model = Discriminator()
target_model.load_state_dict(torch.load("discriminator.pth"))

files = utils.get_file_names("/mnt/data0/fuwenjie/MIA/MIA-Gen/target_model/data/celeba64/total")
all_dataset = Dataset.from_dict({"image": files}).cast_column("image", Image())

datasets = {
    "target": {
        "train": Dataset.from_dict(all_dataset[random.sample(range(0, 50000), cfg["sample_number"])]),
        "valid": Dataset.from_dict(all_dataset[random.sample(range(50000, 60000), cfg["sample_number"])])
            },
}


def norm_transform_images(examples):
    # Preprocessing the datasets and DataLoaders creation.
    # if norm: -1 : 1
    augmentations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )
    images = [augmentations(image.convert("RGB")) for image in examples["image"]]
    return {"input": images}

mem = datasets['target']['train']
mem.set_transform(norm_transform_images)
non_mem = datasets['target']['valid']
non_mem.set_transform(norm_transform_images)
mem = torch.utils.data.DataLoader(mem, batch_size=100)
non_mem = torch.utils.data.DataLoader(non_mem, batch_size=100)

mem_score = []
non_mem_score = []


def eval_attack(y_true, y_scores, plot=True):
    if type(y_true) == torch.Tensor:
        y_true, y_scores = utils.tensor_to_ndarray(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)

    # Finding the threshold point where FPR + TPR equals 1
    threshold_point = tpr[np.argmin(np.abs(tpr - (1 - fpr)))]

    if plot:
        # plot the ROC curve
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score}; ASR = {threshold_point})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        # plot the no-skill line for reference
        plt.plot([0, 1], [0, 1], linestyle='--')
        # show the plot
        plt.show()

for img in mem:
    img = img['input']
    score = target_model(img)
    mem_score.append(score)

for img in non_mem:
    img = img['input']
    score = target_model(img)
    non_mem_score.append(score)

mem_score = torch.cat(mem_score, axis=0)
non_mem_score = torch.cat(non_mem_score, axis=0)
score = torch.cat([mem_score, non_mem_score], axis=0)
score = score.squeeze()
ground_truth = torch.cat([torch.zeros(mem_score.shape[0]), torch.ones(non_mem_score.shape[0])]).type(torch.int).cuda()
eval_attack(ground_truth, -score)

