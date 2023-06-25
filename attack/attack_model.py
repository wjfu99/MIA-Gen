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
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, f1_score
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger.setLevel(logging.INFO)

PATH = os.getcwd()

class MLAttckerModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=2):
        super(MLAttckerModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer(x))
        output = self.output_layer(x)
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

    def target_model_revision(self, model):
        ori_class = model.__class__
        ori_class.loss_function = self.loss_function

    def generative_model_eval(self, model, input, batch_size=100):
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

    def eval_perturb(self, model, dataset, per_num=101, calibration=True):
        """
        Evaluate the loss of the perturbed data

        :param dataset: N*channel*width*height
        :return: losses: N*1; var_losses: N*1; per_losses: N*Mask_Num; ori_losses: N*1
        """
        per_losses = []
        ref_per_losses = []
        model.eval()
        # revising some original methods of target model.
        self.target_model_revision(model)
        ori_losses = self.generative_model_eval(model, dataset)
        if calibration:
            ref_ori_losses = self.generative_model_eval(self.reference_model, dataset)
        for _ in tqdm(range(per_num)):
            per_dataset = self.gaussian_noise_tensor(dataset, 0, 0.1)
            per_loss = self.generative_model_eval(model, per_dataset)
            per_losses.append(per_loss[:, None])
            if calibration:
                ref_per_loss = self.generative_model_eval(self.reference_model, per_dataset)
                ref_per_losses.append(ref_per_loss[:, None])
        per_losses = np.concatenate(per_losses, axis=1)
        var_losses = per_losses - ori_losses[:, None]
        if calibration:
            ref_per_losses = np.concatenate(ref_per_losses, axis=1)
            ref_var_losses = ref_per_losses - ref_ori_losses[:, None]
        if calibration:
            output = (Dict(
                per_losses=per_losses,
                ori_losses=ori_losses,
                var_losses=var_losses,
            ),
            Dict(
                ref_per_losses=ref_per_losses,
                ref_ori_losses=ref_ori_losses,
                ref_var_losses=ref_var_losses,
            ))
        else:
            output = Dict(
                per_losses=per_losses,
                ori_losses=ori_losses,
                var_losses=var_losses,
            )
        return output

    def data_prepare(self, kind, path="attack/attack_data", calibration=True):
        logger.info("Preparing data...")
        data_path = os.path.join(PATH, path)
        target_model = getattr(self, kind + "_model")
        reference_model = self.reference_model
        mem_data = self.datasets[kind]['mem']
        nonmem_data = self.datasets[kind]['nonmem']
        if calibration:
            mem_path = os.path.join(data_path, kind, "cali", "mem_feat.npz")
            nonmem_path = os.path.join(data_path, kind, "cali", "nonmen_feat.npz")
            ref_mem_path = os.path.join(data_path, kind, "cali", "ref_mem_feat.npz")
            ref_nonmem_path = os.path.join(data_path, kind, "cali", "ref_nonmen_feat.npz")
        else:
            mem_path = os.path.join(data_path, kind, "noncali", "mem_feat.npz")
            nonmem_path = os.path.join(data_path, kind, "noncali", "nonmen_feat.npz")
        if not utils.check_files_exist(mem_path, nonmem_path):
            if calibration:
                logger.info("Generating feature vectors for memory data...")
                mem_feat, ref_mem_feat = self.eval_perturb(target_model, mem_data, calibration=calibration)
                # ref_mem_feat = self.eval_perturb(reference_model, mem_data, calibration=calibration)
                logger.info("Generating feature vectors for non-memory data...")
                nonmem_feat, ref_nonmem_feat = self.eval_perturb(target_model, nonmem_data, calibration=calibration)
                # ref_nonmem_feat = self.eval_perturb(reference_model, nonmem_data, calibration=calibration)
                logger.info("Saving feature vectors...")
                utils.save_dict_to_npz(mem_feat, mem_path)
                utils.save_dict_to_npz(nonmem_feat, nonmem_path)
                utils.save_dict_to_npz(ref_mem_feat, ref_mem_path)
                utils.save_dict_to_npz(ref_nonmem_feat, ref_nonmem_path)
            else:
                logger.info("Generating feature vectors for memory data...")
                mem_feat = self.eval_perturb(target_model, mem_data, calibration=calibration)
                logger.info("Generating feature vectors for non-memory data...")
                nonmem_feat = self.eval_perturb(target_model, nonmem_data, calibration=calibration)
                logger.info("Saving feature vectors...")
                utils.save_dict_to_npz(mem_feat, mem_path)
                utils.save_dict_to_npz(nonmem_feat, nonmem_path)
        else:
            if calibration:
                logger.info("Loading feature vectors...")
                mem_feat = utils.load_dict_from_npz(mem_path)
                ref_mem_feat = utils.load_dict_from_npz(ref_mem_path)
                nonmem_feat = utils.load_dict_from_npz(nonmem_path)
                ref_nonmem_feat = utils.load_dict_from_npz(ref_nonmem_path)
            else:
                logger.info("Loading feature vectors...")
                mem_feat = utils.load_dict_from_npz(mem_path)
                nonmem_feat = utils.load_dict_from_npz(nonmem_path)
        logger.info("Data preparation complete.")
        if calibration:
            return Dict(
                mem_feat=mem_feat,
                nonmem_feat=nonmem_feat,
                ref_mem_feat=ref_mem_feat,
                ref_nonmem_feat=ref_nonmem_feat
                        )
        else:
            return Dict(
                mem_feat=mem_feat,
                nonmem_feat=nonmem_feat
            )


    def feat_prepare(self, info_dict):
        # mem_info = info_dict.mem_feat
        # ref_mem_info = info_dict.ref_mem_feat
        mem_feat = info_dict.mem_feat.var_losses / info_dict.mem_feat.ori_losses[:, None]\
                   - info_dict.ref_mem_feat.ref_var_losses / info_dict.ref_mem_feat.ref_ori_losses[:, None]
        nonmem_feat = info_dict.nonmem_feat.var_losses / info_dict.nonmem_feat.ori_losses[:, None]\
                   - info_dict.ref_nonmem_feat.ref_var_losses / info_dict.ref_nonmem_feat.ref_ori_losses[:, None]
        mem_freq = self.frequency(mem_feat, split=100)
        nonmem_freq = self.frequency(nonmem_feat, split=100)
        mem_feat, nonmem_feat = utils.ndarray_to_tensor(mem_freq, nonmem_freq)
        feat = torch.cat([mem_feat, nonmem_feat])
        ground_truth = torch.cat([torch.zeros(mem_feat.shape[0]), torch.ones(nonmem_feat.shape[0])]).type(
            torch.LongTensor).cuda()
        return feat, ground_truth

    def attack_model_training(self, epoch_num=50, load_trained=True):
        # target_model = self.shadow_model
        #
        # mem_data = self.datasets['shadow']['mem']
        # nonmem_data = self.datasets['shadow']['nonmem']
        # mem_dist = self.eval_perturb(target_model, mem_data).per_losses
        # nonmen_dist = self.eval_perturb(target_model, nonmem_data).per_losses
        save_path = os.path.join(PATH, "attack/attack_model", 'attack_model.pth')

        raw_info = self.data_prepare(kind="shadow", calibration=True)
        feat, ground_truth = self.feat_prepare(raw_info)

        eval_raw_info = self.data_prepare(kind="target", calibration=True)
        eval_feat, eval_ground_truth = self.feat_prepare(eval_raw_info)

        feature_dim = feat.shape[-1]
        attack_model = MLAttckerModel(feature_dim).cuda()
        if load_trained and utils.check_files_exist(save_path):
            attack_model.load_state_dict(torch.load(save_path))
            self.attack_model = attack_model
            self.is_model_training = True
            return
        optimizer = optim.Adam(attack_model.parameters(), lr=0.001, weight_decay=0.0005)
        # schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[])
        weight = torch.Tensor([1, 9]).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        print_freq = 10
        for i in range(epoch_num):
            attack_model.train()
            predict = attack_model(feat)
            optimizer.zero_grad()
            loss = criterion(predict, ground_truth)
            loss.backward()
            optimizer.step()
            # Print the loss every 10 epochs
            if i % print_freq == 0:
                attack_model.eval()
                eval_predict = attack_model(eval_feat)
                print(f"Epoch {i} - Loss: {loss.item()}")
                self.eval_attack(ground_truth, predict[:, 1], plot=False)
                self.eval_attack(eval_ground_truth, eval_predict[:, 1], plot=False)
        self.is_model_training = True
        self.attack_model = attack_model
        # Save model
        torch.save(attack_model.state_dict(), save_path)
    def conduct_attack(self, target_samples=None):
        if self.kind == 'ml':
            assert self.is_model_training is True
            attack_model = self.attack_model
            raw_info = self.data_prepare(kind="target", calibration=True)
            feat, ground_truth = self.feat_prepare(raw_info)
            predict = attack_model(feat)
            self.eval_attack(ground_truth, predict[:, 1])
            # dist = self.eval_perturb(self.target_model, target_samples).pre_losses
            # predict = self.ml_model(feat)

    @staticmethod
    def eval_attack(y_true, y_scores, plot=True):
        y_true, y_scores = utils.tensor_to_ndarray(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)
        logger.info(f"Auc on the target model: {auc_score}")
        if plot:
            # plot the ROC curve
            plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc_score)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            # plot the no-skill line for reference
            plt.plot([0, 1], [0, 1], linestyle='--')

            # show the plot
            plt.show()

    @staticmethod
    def gaussian_noise_tensor(tensor, mean=0.0, std=0.1):
        # create a tensor of gaussian noise with the same shape as the input tensor
        noise = torch.randn(tensor.shape) * std + mean
        # add the noise to the original tensor
        noisy_tensor = tensor + noise
        # make sure the pixel values are within [0, 1]
        noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)
        return noisy_tensor

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

    @staticmethod
    def frequency(data, interval=(-1, 1), split=50):
        # Get the number of random variables
        C = data.shape[0]

        # Initialize the frequency vector for each random variable
        freq_vec = np.empty((C, split))

        # Get the range of each random variable
        ranges = np.ptp(data, axis=1)

        # Divide the range of each random variable into N parts
        intervals = np.linspace(interval[0], interval[1], split + 1)

        # Loop through each interval and count the number of occurrences of each random variable
        for i in range(C):
            for j in range(split):
                freq_vec[i][j] = len(
                    np.where(np.logical_and(data[i] >= intervals[j], data[i] <= intervals[j + 1]))[0])

        return freq_vec