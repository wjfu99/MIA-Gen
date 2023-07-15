import os
import numpy as np
import torch
import pythae
import torch.nn.functional as F
import logging

from attack.attack_model_diffusion import AttackModel

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

target_path = os.path.join(PATH, 'diffusion_models/ddpm-celeba-64-50k/checkpoint-36000')
target_model = DiffusionPipeline.from_pretrained(target_path).to(device)


shadow_path = os.path.join(PATH, 'diffusion_models/ddpm-celeba-64-shadow')
shadow_model = None

reference_path = os.path.join(PATH, 'diffusion_models/ddpm-celeba-64-reference')
reference_model = None

logger.info("Successfully loaded models!")

# Load datasets
# splits = []
# for model_type in ["target", "shadow", "reference"]:
#     for data_type in ["train", "valid"]:
#         splits.append(model_type+"_"+data_type)

files = utils.get_file_names("/mnt/data0/fuwenjie/MIA/MIA-Gen/target_model/data/celeba64/total")
all_dataset = Dataset.from_dict({"image": files}).cast_column("image", Image())




datasets = {
    "target": {
        "train": Dataset.from_dict(all_dataset[0:100]),
        "valid": Dataset.from_dict(all_dataset[150000:150100])
            },
    "shadow": {
        "train": Dataset.from_dict(all_dataset[100000:100100]),
        "valid": Dataset.from_dict(all_dataset[110000:110100])
    },
    "reference": {
        "train": Dataset.from_dict(all_dataset[150000:150100]),
        "valid": Dataset.from_dict(all_dataset[160000:160100])
    }
}

attack_model = AttackModel(target_model, datasets, reference_model, shadow_model, cfg=cfg)
# attack_model.attack_demo(cfg, target_model)
# attack_model.attack_model_training(cfg=cfg)
attack_model.conduct_attack(cfg=cfg)