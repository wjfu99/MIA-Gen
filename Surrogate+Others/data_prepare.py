import os
import numpy as np
import torch
import pythae
import torch.nn.functional as F
import logging
import random
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
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

PATH = os.path.dirname(os.path.abspath(__file__))
# Automatically select the freest GPU.
os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_available))
device = "cuda" + ":" + str(np.argmax(memory_available))
torch.cuda.set_device(device)

cfg = {
    "target_model": "diffusion",
    "dataset": "celeba",
    "sample_num": 50000

}


if cfg["target_model"] == "diffusion":
    target_path = os.path.join(PATH, '../diffusion_models/ddpm-celeba-64-50k/checkpoint-247500')
    target_model = DiffusionPipeline.from_pretrained(target_path).to(device)

elif cfg["target_model"] == "vae":
    target_path = sorted(os.listdir(os.path.join(PATH, '../target_model/target_models_on_' + cfg["dataset"] + "_50k")))[-1]
    target_model = AutoModel.load_from_folder(
        os.path.join(PATH, '../target_model/target_models_on_' + cfg["dataset"] + "_50k", target_path, 'final_model'))
    target_model = target_model.to(device)

def gen_data_vae(model, img_path, sample_numbers=3000, batch_size=100):  # TODO: sample without the fit function.
    model.eval()
    z_dim = model.latent_dim
    generated_samples = []
    img_idx = 0
    with torch.no_grad():
        for i in tqdm(range(0, sample_numbers, batch_size)):
            z = torch.normal(0, 1, size=(batch_size, z_dim)).cuda()
            gen = model.decoder(z)["reconstruction"]
            # gen = utils.tensor_to_ndarray(gen)[0]
            utils.create_folder(img_path)
            for img in gen:
                img = ToPILImage()(img)
                img.save(img_path + f"/{img_idx}.jpg")
                img_idx += 1

def gen_data_diffusion(model, img_path, sample_numbers=100, batch_size=100):
    pipeline = model
    generated_samples = []
    img_idx = 0
    utils.create_folder(img_path)
    for i in tqdm(range(0, sample_numbers, batch_size)):
        gen = pipeline(batch_size).images
        for i, img in enumerate(gen):
            img.save(img_path + f"/{img_idx}.jpg")
            img_idx += 1


img_path = os.path.normpath(os.path.join(PATH, f"../data/gen_dataset/{cfg['target_model']}@{cfg['dataset']}"))
if cfg["target_model"] == "diffusion":
    gen_data_diffusion(target_model, img_path, sample_numbers=cfg['sample_num'])
elif cfg["target_model"] == "vae":
    gen_data_vae(target_model, img_path, sample_numbers=cfg['sample_num'])
