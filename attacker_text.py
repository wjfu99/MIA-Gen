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
with open("/mnt/data0/fuwenjie/MIA/MIA-Gen/SentenceVAE/data" + '/ptb.vocab.json', 'r') as file:
    vocab = json.load(file)
w2i, i2w = vocab['w2i'], vocab['i2w']

target_path = os.path.join(PATH, 'SentenceVAE/bin/target')
target_path = os.path.join(target_path, sorted(os.listdir(target_path))[-1])
with open(target_path + "/model_params.json", 'r') as file:
    params = json.load(file)
target_model = SentenceVAE(**params).to("cuda")
target_model.load_state_dict(torch.load(target_path+"/E9.pytorch"))

shadow_path = os.path.join(PATH, 'SentenceVAE/bin/shadow')
shadow_path = os.path.join(shadow_path, sorted(os.listdir(shadow_path))[-1])
with open(shadow_path + "/model_params.json", 'r') as file:
    params = json.load(file)
shadow_model = SentenceVAE(**params).to("cuda")
shadow_model.load_state_dict(torch.load(shadow_path+"/E9.pytorch"))

reference_path = os.path.join(PATH, 'SentenceVAE/bin/reference')
reference_path = os.path.join(reference_path, sorted(os.listdir(reference_path))[-1])
with open(reference_path + "/model_params.json", 'r') as file:
    params = json.load(file)
reference_model = SentenceVAE(**params).to("cuda")
reference_model.load_state_dict(torch.load(reference_path+"/E9.pytorch"))

logger.info("Successfully loaded models!")

# Load datasets

model_type = "reference"
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