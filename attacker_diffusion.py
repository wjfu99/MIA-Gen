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
from data.prepare import data_prepare

# Load config file
with open("configs/config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

# Add Logger
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger.setLevel(logging.INFO)

# Load abs path
PATH = os.path.dirname(os.path.abspath(__file__))

# Automatically select the freest GPU.
os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_available))
device = "cuda" + ":" + str(np.argmax(memory_available))
torch.cuda.set_device(device)

# Fix the random seed
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


## Load generation models.
if cfg['dataset'] == "tinyin":
    if cfg["target_model"] == "diffusion":
        target_path = os.path.join(PATH, 'diffusion_models/ddpm-tinyin-64-30k')
        target_model = DiffusionPipeline.from_pretrained(target_path).to(device)

        # shadow_path = os.path.join(PATH, 'diffusion_models/ddpm-celeba-64-50k-shadow/checkpoint-247500')
        # shadow_model = DiffusionPipeline.from_pretrained(shadow_path).to(device)
        shadow_model = None

        # reference_path = os.path.join(PATH, 'diffusion_models/ddpm-celeba-64-50k-reference/checkpoint-247500')
        # reference_model = DiffusionPipeline.from_pretrained(reference_path).to(device)
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
elif cfg['dataset'] == "celeba":
    if cfg["target_model"] == "diffusion":
        target_path = os.path.join(PATH, 'diffusion_models/ddpm-celeba-64-50k/checkpoint-247500')
        target_model = DiffusionPipeline.from_pretrained(target_path).to(device)

        shadow_path = os.path.join(PATH, 'diffusion_models/ddpm-celeba-64-50k-shadow/checkpoint-247500')
        shadow_model = DiffusionPipeline.from_pretrained(shadow_path).to(device)

        reference_path = os.path.join(PATH, 'diffusion_models/ddpm-celeba-64-50k-reference/checkpoint-247500')
        reference_model = DiffusionPipeline.from_pretrained(reference_path).to(device)
    elif cfg["target_model"] == "vae":
        target_path = sorted(os.listdir(PATH + '/target_model/target_models_on_' + cfg["dataset"] + "_50k"))[-1]
        target_model = AutoModel.load_from_folder(
            os.path.join(PATH + '/target_model/target_models_on_' + cfg["dataset"] + "_50k", target_path,
                         'final_model'))
        target_model = target_model.to(device)

        reference_path = sorted(os.listdir(PATH + '/target_model/reference_models_on_' + cfg["dataset"] + "_50k"))[-1]
        reference_model = AutoModel.load_from_folder(
            os.path.join(PATH + '/target_model/reference_models_on_' + cfg["dataset"] + "_50k", reference_path,
                         'final_model'))
        reference_model = reference_model.to(device)

        shadow_path = sorted(os.listdir(PATH + '/target_model/shadow_models_on_' + cfg["dataset"] + "_50k"))[-1]
        shadow_model = AutoModel.load_from_folder(
            os.path.join(PATH + '/target_model/shadow_models_on_' + cfg["dataset"] + "_50k", shadow_path,
                         'final_model'))
        shadow_model = shadow_model.to(device)

logger.info("Successfully loaded models!")

# Load datasets
all_dataset = data_prepare(cfg['dataset'], mode="datasets")

if cfg['dataset'] == "tinyin":
    datasets = {
        "target": {
            "train": Dataset.from_dict(all_dataset[random.sample(range(0, 30000), cfg["sample_number"])]),
            "valid": Dataset.from_dict(all_dataset[random.sample(range(30000, 35000), cfg["sample_number"])])
                },
        "shadow": {
            "train": Dataset.from_dict(all_dataset[random.sample(range(35000, 65000), cfg["sample_number"])]),
            "valid": Dataset.from_dict(all_dataset[random.sample(range(65000, 70000), cfg["sample_number"])])
        },
        "reference": {
            "train": Dataset.from_dict(all_dataset[random.sample(range(70000, 100000), cfg["sample_number"])]),
            "valid": Dataset.from_dict(all_dataset[random.sample(range(100000, 105000), cfg["sample_number"])])
        }
    }
elif cfg['dataset'] == "celeba":
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