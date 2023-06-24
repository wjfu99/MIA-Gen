import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import sklearn.metrics as metrics
import numpy as np
import pickle as pkl
from tqdm import tqdm
from attack import utils
from attack.utils import Dict

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger.setLevel(logging.INFO)

PATH = os.getcwd()

class MLAttckerModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=3):
        super(MLAttckerModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer(x))
        output = torch.softmax(self.output_layer(x), dim=1)
        return output


class AttackModel:
    def __init__(self, target_model, datasets, reference_model=None, shadow_model=None, kind="ml"):
        self.target_model = target_model
        self.datasets = datasets
        self.kind = kind
        if shadow_model is not None and kind == "ml":
            self.shadow_model = shadow_model
            self.is_model_training = False
        if reference_model is not None:
            self.reference_model = reference_model
    @staticmethod
    def output_reformat(output_dict):
        for key in output_dict.keys():
            output_dict[key] = output_dict[key].cpu().detach().numpy()
        return output_dict
    @staticmethod
    def loss_function(self, recon_x, x, mu, log_var, z):

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        return (recon_loss + KLD), recon_loss, KLD

    def target_model_revision(self, model):
        ori_class = model.__class__
        ori_class.loss_function = self.loss_function

    def generative_model_eval(self, model, input, batch_size=1000):
        input = input.cuda()
        num_inputs = input.shape[0]
        outputs = []
        model.eval()

        for i in range(0, num_inputs, batch_size):
            input_batch = input[i:i + batch_size]
            input_dict = {"data": input_batch}
            output_batch = self.output_reformat(model(input_dict)).loss
            outputs.append(output_batch)
        output = np.concatenate(outputs, axis=0)
        return output

    def eval_perturb(self, model, dataset, per_num=101):
        """
        Evaluate the loss of the perturbed data

        :param dataset: N*channel*width*height
        :return: losses: N*1; var_losses: N*1; per_losses: N*Mask_Num; ori_losses: N*1
        """
        losses = []
        ori_losses = []
        var_losses = []
        per_losses = []
        peak_losses = []
        model.eval()
        # revising some original methods of target model.
        self.target_model_revision(model)
        ori_losses = self.generative_model_eval(model, dataset)
        for _ in tqdm(range(per_num)):
            per_dataset = self.gaussian_noise_tensor(dataset, 0, 0.1)
            per_loss = self.generative_model_eval(model, per_dataset)
            per_losses.append(per_loss[:, None])
        per_losses = np.concatenate(per_losses, axis=1)
        var_losses = per_losses - ori_losses[:, None]
        # for data in tqdm(dataset):
        #     data = torch.unsqueeze(data, 0)
        #     # show_image(data[0])
        #     # show_image(eval_loss(data).recon_x.detach()[0])
        #     ori_loss = self.generative_model_eval(model, data).loss.item()
        #     # masks = mask_tensor(data, prob=0.05, num_masks=per_num)
        #     masks = self.gaussian_noise_tensor(data, 0, 0.1, per_num)
        #     # masks = add_gaussian_noise(data, noise_scale=0.1, num_noised=per_num)
        #     per_loss = []
        #     avg_loss = 0
        #     for mask in masks:
        #         mask = torch.unsqueeze(mask, 0)
        #         recon_loss = self.generative_model_eval(model, mask).loss.item()
        #         per_loss.append(recon_loss)
        #         avg_loss += recon_loss
        #     avg_loss = avg_loss / per_num
        #     ori_losses.append(ori_loss)
        #     per_losses.append(per_loss)
        #     losses.append(avg_loss)
        #     var_losses.append(avg_loss - ori_loss)
        #     peak_losses.append(np.min(per_loss) - ori_loss)
        output = Dict(
            per_losses=per_losses,
            ori_losses=ori_losses,
            var_losses=var_losses,
            # losses=np.array(losses),
            # peak_losses=np.array(peak_losses)
        )
        return output

    def data_prepare(self, path="attack/attack_data", type="shadow"):
        logger.info("Preparing data...")
        data_path = os.path.join(PATH, path)
        target_model = getattr(self, type+"_model")
        mem_data = self.datasets[type]['mem']
        nonmem_data = self.datasets[type]['nonmem']
        mem_path = os.path.join(data_path, type, "mem_feat.pkl")
        nonmem_path = os.path.join(data_path, type,"nonmen_feat.pkl")
        if not utils.check_files_exist(mem_path, nonmem_path):
            logger.info("Generating feature vectors for memory data...")
            mem_feat = self.eval_perturb(target_model, mem_data)
            logger.info("Generating feature vectors for non-memory data...")
            nonmem_feat = self.eval_perturb(target_model, nonmem_data)
            logger.info("Saving feature vectors...")
            utils.save_dict_to_npz(mem_feat, mem_path)
            utils.save_dict_to_npz(nonmem_feat, nonmem_path)
        else:
            logger.info("Loading feature vectors...")
            mem_feat = utils.load_dict_from_npz(mem_path)
            nonmem_feat = utils.load_dict_from_npz(nonmem_path)
        logger.info("Data preparation complete.")
        return mem_feat, nonmem_feat

    def attack_model_training(self, epoch_num=100):
        # target_model = self.shadow_model
        #
        # mem_data = self.datasets['shadow']['mem']
        # nonmem_data = self.datasets['shadow']['nonmem']
        # mem_dist = self.eval_perturb(target_model, mem_data).per_losses
        # nonmen_dist = self.eval_perturb(target_model, nonmem_data).per_losses
        mem_feat, nonmem_feat = self.data_prepare()
        mem_feat, nonmem_feat = mem_feat.var_losses, nonmem_feat.var_losses
        mem_feat, nonmem_feat = utils.ndarray_to_tensor(mem_feat, nonmem_feat)
        feat = np.concatenate(mem_feat, nonmem_feat)
        ground_truth = torch.cat([torch.zeros(mem_feat.shape[0]), torch.ones(nonmem_feat.shape[0])]).cuda()
        feature_dim = mem_feat.shape[-1]
        attack_model = MLAttckerModel(feature_dim).cuda()
        optimizer = optim.Adam(attack_model.parameters(), lr=0.001, weight_decay=0.0005)
        # schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[])
        criterion = torch.nn.CrossEntropyLoss()
        print_freq = 10
        for i in range(epoch_num):
            for phase in ['train', 'val']:
                if phase == 'train':
                    attack_model.train()
                    predict = attack_model(feat)
                else:
                    attack_model.eval()
                    predict = attack_model(nonmem_feat)
                optimizer.zero_grad()
                loss = criterion(predict, ground_truth)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
        self.is_model_training = True

        # Save model
        save_path = os.path.join(PATH, "attack/attack_model", 'attack_model.pth')
        torch.save(attack_model.state_dict(), save_path)
    def conduct_attack(self, target_samples):
        if self.kind == 'ml':
            assert self.is_model_training is True
            dist = self.eval_perturb(self.target_model, target_samples).pre_losses
            predict = self.ml_model(dist)
    def eval_attack(self, predict, true):
        attack_model = self.ml_model
        attack_model()

    @staticmethod
    def gaussian_noise_tensor(tensor, mean=0.0, std=0.1):
        # create a tensor of gaussian noise with the same shape as the input tensor
        noise = torch.randn(tensor.shape) * std + mean
        # add the noise to the original tensor
        noisy_tensor = tensor + noise
        # make sure the pixel values are within [0, 1]
        noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)
        return noisy_tensor