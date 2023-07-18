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

with open("configs/config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))
# Automatically select the freest GPU.
os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_available))
device = "cuda" + ":" + str(np.argmax(memory_available))
torch.cuda.set_device(device)


## Load text generation model.
if cfg["target_model"] == "diffusion":
    target_path = os.path.join(PATH, 'diffusion_models/ddpm-celeba-64-50k/checkpoint-247500')
    target_model = DiffusionPipeline.from_pretrained(target_path).to(device)


    shadow_path = os.path.join(PATH, 'diffusion_models/ddpm-celeba-64-shadow')
    shadow_model = None

    reference_path = os.path.join(PATH, 'diffusion_models/ddpm-celeba-64-reference')
    reference_model = None
elif cfg["target_model"] == "vae":
    target_path = sorted(os.listdir(PATH + '/target_model/target_models_on_' + cfg["dataset"] + "_50k"))[-1]
    target_model = AutoModel.load_from_folder(
        os.path.join(PATH + '/target_model/target_models_on_' + cfg["dataset"] + "_50k", target_path, 'final_model'))
    target_model = target_model.to(device)

    reference_path = sorted(os.listdir(PATH + '/target_model/reference_models_on_' + cfg["dataset"] + "_50k"))[-1]
    reference_model = AutoModel.load_from_folder(
        os.path.join(PATH + '/target_model/reference_models_on_' + cfg["dataset"] + "_50k", reference_path, 'final_model'))
    reference_model = reference_model.to(device)

    shadow_path = sorted(os.listdir(PATH + '/target_model/shadow_models_on_' + cfg["dataset"] + "_50k"))[-1]
    shadow_model = AutoModel.load_from_folder(
        os.path.join(PATH + '/target_model/shadow_models_on_' + cfg["dataset"] + "_50k", shadow_path, 'final_model'))
    shadow_model = shadow_model.to(device)

logger.info("Successfully loaded models!")

# Load datasets
# splits = []
# for model_type in ["target", "shadow", "reference"]:
#     for data_type in ["train", "valid"]:
#         splits.append(model_type+"_"+data_type)

files = utils.get_file_names("/mnt/data0/fuwenjie/MIA/MIA-Gen/target_model/data/celeba64/total")
all_dataset = Dataset.from_dict({"image": files}).cast_column("image", Image())


# data_split = {
#     "target": {
#         "train": (0, 50000),
#         "valid": (50000, 60000)
#     },
#     "shadow": {
#         "train": (60000, 110000),
#         "valid": (110000, 120000)
#     },
#     "reference": {
#         "train": (120000, 170000),
#         "valid": (170000, 180000)
#     }
# }

datasets = {
    "target": {
        "train": Dataset.from_dict(all_dataset[random.sample(range(0, 50000), cfg["sample_number"])]),
        "valid": Dataset.from_dict(all_dataset[random.sample(range(50000, 60000), cfg["sample_number"])])
            },
    "shadow": {
        "train": Dataset.from_dict(all_dataset[random.sample(range(60000, 110000), cfg["sample_number"])]),
        "valid": Dataset.from_dict(all_dataset[random.sample(range(110000, 120000), cfg["sample_number"])])
    },
    "reference": {
        "train": Dataset.from_dict(all_dataset[random.sample(range(120000, 170000), cfg["sample_number"])]),
        "valid": Dataset.from_dict(all_dataset[random.sample(range(170000, 180000), cfg["sample_number"])])
    }
}

attack_model = AttackModel(target_model, datasets, reference_model, shadow_model, cfg=cfg)
# attack_model.attack_demo(cfg, target_model)
# attack_model.attack_model_training(cfg=cfg)
attack_model.conduct_attack(cfg=cfg)