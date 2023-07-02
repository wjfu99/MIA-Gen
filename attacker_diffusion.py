import os
import numpy as np
import torch
import pythae
import torch.nn.functional as F
import logging
from imageio import imwrite
from pythae.models import AutoModel
from pythae.samplers import NormalSampler
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, f1_score
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from attack.attack_model_text import AttackModel
from SentenceVAE.model import SentenceVAE
from SentenceVAE.ptb import PTB
from diffusers import DiffusionPipeline
from collections import OrderedDict
import json

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
device = "cuda" if torch.cuda.is_available() else "cpu"


## Load text generation model.

target_path = os.path.join(PATH, 'diffusion_models/ddpm-celeba-64-target')
target_model = DiffusionPipeline.from_pretrained(target_path)


shadow_path = os.path.join(PATH, 'diffusion_models/ddpm-celeba-64-shadow')
shadow_model = DiffusionPipeline.from_pretrained(shadow_path)

reference_path = os.path.join(PATH, 'diffusion_models/ddpm-celeba-64-reference')
reference_model = DiffusionPipeline.from_pretrained(reference_path)

logger.info("Successfully loaded models!")

# Load datasets
splits = []
for model_type in ["target", "shadow", "reference"]:
    for data_type in ["train", "valid"]:
        splits.append(model_type+"_"+data_type)

datasets = OrderedDict()
for split in splits:
    datasets[split] = PTB(
        data_dir="SentenceVAE/data",
        split=split,
        create_data=False,
        max_sequence_length=60,
        min_occ=1
    )

attack_model = AttackModel(target_model, datasets, reference_model, shadow_model)
attack_model.attack_model_training()
attack_model.conduct_attack()