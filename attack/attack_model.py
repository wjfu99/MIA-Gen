import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import sklearn.metrics as metrics
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger.setLevel(logging.INFO)


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

class OutputDict(dict):
    def __getattr__(self, name):
        if name in self:
            return  self[name]
        raise AttributeError(f"'OutputDict' object has no attribute '{name}'")
    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)
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

        for i in tqdm(range(0, num_inputs, batch_size)):
            input_batch = input[i:i + batch_size]
            input_dict = {"data": input_batch}
            output_batch = self.output_reformat(model(input_dict)).loss
            outputs.append(output_batch)
        output = np.concatenate(outputs, axis=0)
        return output

    def eval_perturb(self, model, dataset):
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
        per_num = 100
        model.eval()
        # revising some original methods of target model.
        self.target_model_revision(model)
        ori_losses = self.generative_model_eval(model, dataset)
        for data in tqdm(dataset):
            data = torch.unsqueeze(data, 0)
            # show_image(data[0])
            # show_image(eval_loss(data).recon_x.detach()[0])
            ori_loss = self.generative_model_eval(model, data).loss.item()
            # masks = mask_tensor(data, prob=0.05, num_masks=per_num)
            masks = self.gaussian_noise_tensor(data, 0, 0.1, per_num)
            # masks = add_gaussian_noise(data, noise_scale=0.1, num_noised=per_num)
            per_loss = []
            avg_loss = 0
            for mask in masks:
                mask = torch.unsqueeze(mask, 0)
                recon_loss = self.generative_model_eval(model, mask).loss.item()
                per_loss.append(recon_loss)
                avg_loss += recon_loss
            avg_loss = avg_loss / per_num
            ori_losses.append(ori_loss)
            per_losses.append(per_loss)
            losses.append(avg_loss)
            var_losses.append(avg_loss - ori_loss)
            peak_losses.append(np.min(per_loss) - ori_loss)
        output = OutputDict(
            per_losses = np.array(per_losses),
            ori_losses = np.array(ori_losses),
            losses = np.array(losses),
            var_losses = np.array(var_losses),
            peak_losses = np.array(peak_losses)
        )
        return output

    def attack_model_training(self, epoch_num=100):
        target_model = self.shadow_model

        mem_data = self.datasets['shadow']['mem']
        nonmem_data = self.datasets['shadow']['nonmem']
        mem_dist = self.eval_perturb(target_model, mem_data).per_losses
        nonmen_dist = self.eval_perturb(target_model, nonmem_data).per_losses
        ground_truth = torch.concat([torch.zeros(nonmem_data[0]), torch.ones(mem_data.shape[0])]).cuda()
        feature_dim = mem_dist.shape[-1]
        attack_model = MLAttckerModel(feature_dim)
        optimizer = optim.Adam(attack_model.parameters(), lr=0.001, weight_decay=0.0005)
        # schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[])
        criterion = torch.nn.CrossEntropyLoss()
        print_freq = 10
        for i in range(epoch_num):
            for phase in ['train', 'val']:
                if phase == 'train':
                    attack_model.train()
                    predict = attack_model(mem_dist)
                else:
                    attack_model.eval()
                    predict = attack_model(nonmen_dist)
                optimizer.zero_grad()
                loss = criterion(predict, ground_truth)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
        self.is_model_training = True
    def conduct_attack(self, target_samples):
        if self.kind == 'ml':
            assert self.is_model_training is True
            dist = self.eval_perturb(self.target_model, target_samples).pre_losses
            predict = self.ml_model(dist)
    def eval_attack(self, predict, true):
        attack_model = self.ml_model
        attack_model()

    @staticmethod
    def gaussian_noise_tensor(tensor, mean=0.0, std=0.1, num_noise=1):
        noises = []
        for i in range(num_noise):
            # create a tensor of gaussian noise with the same shape as the input tensor
            noise = torch.randn(tensor.shape) * std + mean
            # add the noise to the original tensor
            noisy_tensor = tensor + noise
            # make sure the pixel values are within [0, 1]
            noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)
            noises.append(noisy_tensor)
        noises = torch.stack(noises, dim=0)
        noises = torch.squeeze(noises)
        return noises