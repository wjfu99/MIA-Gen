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
from attack.attack_model import AttackModel

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



dataset = "celeba"

target_path = sorted(os.listdir(PATH + '/target_model/target_models_on_' + dataset))[-1]
target_model = AutoModel.load_from_folder(os.path.join(PATH + '/target_model/target_models_on_'+dataset, target_path, 'final_model'))
target_model = target_model.to(device)

reference_path = sorted(os.listdir(PATH + '/target_model/reference_models_on_'+dataset))[-1]
reference_model = AutoModel.load_from_folder(os.path.join(PATH + '/target_model/reference_models_on_'+dataset, reference_path, 'final_model'))
reference_model = reference_model.to(device)

shadow_path = sorted(os.listdir(PATH + '/target_model/shadow_models_on_'+dataset))[-1]
shadow_model = AutoModel.load_from_folder(os.path.join(PATH + '/target_model/shadow_models_on_'+dataset, shadow_path, 'final_model'))
shadow_model = shadow_model.to(device)
logger.info("Successfully loaded models!")
# last_training = sorted(os.listdir('target_model/my_model'))[-1]
# trained_model = AutoModel.load_from_folder(os.path.join('target_model/my_model', last_training, 'final_model'))
# trained_model = trained_model.to(device)
if dataset == "celeba":
    celeba64_dataset = np.load("./target_model/data/celeba64/celeba64.npz")["arr_0"] / 255.0
    datasets = dict(
        target=dict(
            mem=torch.Tensor(celeba64_dataset[0:10000]),
            nonmem=torch.Tensor(celeba64_dataset[10000:13000])
        ),
        shadow=dict(
            mem=torch.Tensor(celeba64_dataset[100000:110000]),
            nonmem=torch.Tensor(celeba64_dataset[110000:113000])
        ),
        reference=dict(
            mem=torch.Tensor(celeba64_dataset[150000:160000]),
            nonmem=torch.Tensor(celeba64_dataset[160000:163000])
        )
    )
    logger.info("Successfully loaded datasets!")
else:
    logger.info(f"\nLoading {dataset} data...\n")
    train_data = torch.Tensor(
            np.load(os.path.join(PATH, f"target_model/data/{dataset}", "train_data.npz"))[
                "data"
            ]
            / 255.0
    )
    if dataset == "mnist":
        train_data = train_data[:-10000]
    eval_data = torch.Tensor(
            np.load(os.path.join(PATH, f"target_model/data/{dataset}", "eval_data.npz"))["data"]
            / 255.0
    )

attack_model = AttackModel(target_model, datasets, reference_model, shadow_model)
attack_model.attack_model_training()
attack_model.conduct_attack()
