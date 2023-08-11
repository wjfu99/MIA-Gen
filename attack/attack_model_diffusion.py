import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import numpy as np
import pickle as pkl
from tqdm import tqdm
from attack import utils
from attack.utils import Dict
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, f1_score
from itertools import cycle
import matplotlib.pyplot as plt
from copy import deepcopy, copy
import random
import time
from torchvision import transforms
from datasets import Image, Dataset
from attack.resnet import ResNet18
import seaborn as sns

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
logger.setLevel(logging.INFO)

PATH = os.getcwd()

NLL = torch.nn.NLLLoss(ignore_index=0, reduction='none')

# class MLAttckerModel(nn.Module):
#     def __init__(self, input_size, hidden_size=128, output_size=2):
#         super(MLAttckerModel, self).__init__()
#         self.input_layer = nn.Linear(input_size, hidden_size)
#         self.hidden_layer = nn.Linear(hidden_size, hidden_size)
#         self.output_layer = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         x = torch.relu(self.input_layer(x))
#         x = torch.relu(self.hidden_layer(x))
#         output = self.output_layer(x)
#         return output

class MLAttckerModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=2):
        super(MLAttckerModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(3, 8, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(8, 4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(4, 1, kernel_size=3, stride=1, padding=1)

        self.output_layer = nn.Linear(5, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        output = self.output_layer(x)
        output = output.squeeze(1)
        return output

class AttackModel:
    def __init__(self, target_model, datasets, reference_model, shadow_model, cfg):
        if cfg["target_model"] == "vae":
            self.target_model_revision(target_model)
        self.target_model = target_model
        self.datasets = datasets
        self.kind = cfg['attack_kind']
        if shadow_model is not None and cfg['attack_kind'] == "nn":
            self.shadow_model = shadow_model
            self.is_model_training = False
        if reference_model is not None:
            self.reference_model = reference_model

    def target_model_revision(self, model):
        ori_class = model.__class__
        ori_class.loss_function = self.loss_function

    @staticmethod
    def loss_fn(logp, target, length, mean, logv):
        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)
        NLL_loss = NLL_loss.reshape(100, -1).sum(-1) # TODO: The loss function may should not use sum.
        # KL Divergence
        KL_loss = -0.5 * (1 + logv - mean.pow(2) - logv.exp()).sum(-1)
        KL_weight = 0.5 # TODO: an variational value.

        loss = NLL_loss + KL_loss * KL_weight

        return loss
    @staticmethod
    def ddpm_loss(pipeline, clean_images, timestep):
        model = pipeline.unet
        model.eval()
        noise_scheduler = pipeline.scheduler
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        timesteps = torch.full((clean_images.shape[0],), timestep, device=clean_images.device)
        noise_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        model_output = model(noise_images, timesteps).sample
        loss = F.mse_loss(model_output, noise, reduction="none")
        loss = torch.mean(loss, dim=(1, 2, 3))
        loss = utils.tensor_to_ndarray(loss)[0]
        return loss


    @staticmethod
    def ddim_singlestep(pipeline, x, t_c, t_target):
        model = pipeline.unet
        model.eval()
        noise_scheduler = pipeline.scheduler
        x = x.cuda()

        t_c = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_c)
        t_target = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_target)

        # betas = torch.linspace(FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).double().cuda()
        betas = noise_scheduler.betas.cuda()
        alphas = 1. - betas
        alphas = torch.cumprod(alphas, dim=0)

        alphas_t_c = utils.extract(alphas, t=t_c, x_shape=x.shape)
        alphas_t_target = utils.extract(alphas, t=t_target, x_shape=x.shape)

        with torch.no_grad():
            epsilon = model(x, t_c).sample

        pred_x_0 = (x - ((1 - alphas_t_c).sqrt() * epsilon)) / (alphas_t_c.sqrt())
        x_t_target = alphas_t_target.sqrt() * pred_x_0 \
                     + (1 - alphas_t_target).sqrt() * epsilon

        return {
            'x_t_target': x_t_target,
            'epsilon': epsilon
        }

    def ddim_multistep(self, pipeline, x, t_c, target_steps, clip=False):
        for idx, t_target in enumerate(target_steps):
            result = self.ddim_singlestep(pipeline, x, t_c, t_target)
            x = result['x_t_target']
            t_c = t_target

        if clip:
            result['x_t_target'] = torch.clip(result['x_t_target'], -1, 1)

        return result

    def ddim_loss(self, model, x, timestep=10, t_sec=100):
        target_steps = list(range(0, t_sec, timestep))[1:]
        x_sec = self.ddim_multistep(model, x, t_c=0, target_steps=target_steps)
        x_sec = x_sec['x_t_target']
        x_sec_recon = self.ddim_singlestep(model, x_sec, t_c=target_steps[-1], t_target=target_steps[-1] + timestep)
        x_sec_recon = self.ddim_singlestep(model, x_sec_recon['x_t_target'], t_c=target_steps[-1] + timestep,
                                           t_target=target_steps[-1])
        x_sec_recon = x_sec_recon['x_t_target']
        loss = (x_sec_recon - x_sec) ** 2
        loss = loss.flatten(1).sum(dim=-1)
        loss = utils.tensor_to_ndarray(loss)[0]
        return loss

    def diffusion_eval(self, model, input, cfg):
        outputs = []
        data_loader = DataLoader(
            dataset=input,
            batch_size=cfg["eval_batch_size"],
            shuffle=False,
            num_workers=32,
            pin_memory=torch.cuda.is_available()
        )
        pipeline = model
        model = pipeline.unet
        model.eval()
        noise_scheduler = pipeline.scheduler
        loss_function = getattr(self, cfg["loss_kind"]+"_loss")
        diffusion_steps = noise_scheduler.config.num_train_timesteps
        interval = diffusion_steps // cfg["diffusion_sample_number"]
        sample_steps = cfg["diffusion_sample_steps"]

        for iteration, batch in enumerate(data_loader):
            clean_images = batch["input"].cuda()
            batch_loss = np.zeros((cfg["eval_batch_size"], cfg["diffusion_sample_number"]))
            # start_time = time.time()
            for i, timestep in enumerate(sample_steps):
                if cfg["loss_kind"] == "ddpm":
                    loss = self.ddpm_loss(pipeline, clean_images, timestep)
                elif cfg["loss_kind"] == "ddim":
                    loss = self.ddim_loss(pipeline, clean_images, t_sec=timestep)
                batch_loss[:, i] = loss
            # print(f"time duration: {time.time() - start_time}s")
            outputs.append(batch_loss)
        output = np.concatenate(outputs, axis=0)
        return output

    def vae_eval(self, model, input, cfg):
        outputs = []
        data_loader = DataLoader(
            dataset=input,
            batch_size=cfg["eval_batch_size"],
            shuffle=False,
            num_workers=32,
            pin_memory=torch.cuda.is_available()
        )
        for iteration, batch in enumerate(data_loader):
            batch_loss = np.zeros((cfg["eval_batch_size"], cfg["extensive_per_num"]))
            for i in range(cfg['extensive_per_num']):
                input_dict = {"data": batch["input"].cuda()}
                output_batch = self.output_reformat(model(input_dict)).loss
                batch_loss[:, i] = output_batch
            outputs.append(batch_loss)
        output = np.concatenate(outputs, axis=0)
        # input = input.cuda()
        # num_inputs = input.shape[0]
        # outputs = []
        # model.eval()
        #
        # for i in range(0, num_inputs, batch_size):
        #     input_batch = input[i:i + batch_size]
        #     input_dict = {"data": input_batch}
        #     output_batch = self.output_reformat(model(input_dict)).loss
        #     outputs.append(output_batch)
        # output = np.concatenate(outputs, axis=0)
        return output

    def eval_perturb(self, model, dataset, cfg):
        """
        Evaluate the loss of the perturbed data

        :param dataset: N*channel*width*height
        :return: losses: N*1; var_losses: N*1; per_losses: N*Mask_Num; ori_losses: N*1
        """
        per_losses = []
        ref_per_losses = []
        # revising some original methods of target model.
        # self.target_model_revision(model)
        model_eval_function = getattr(self, cfg["target_model"] + "_eval")
        dataset_perturbation_function = self.image_dataset_perturbation if cfg["target_model"] == "vae" else self.norm_image_dataset_perturbation
        ori_dataset = deepcopy(dataset)
        ori_dataset.set_transform(self.transform_images if cfg["target_model"] == "vae" else self.norm_transform_images)
        ori_losses = model_eval_function(model, ori_dataset, cfg)
        ref_ori_losses = model_eval_function(self.reference_model, ori_dataset, cfg) if cfg["calibration"] else None
        strength = np.linspace(cfg['start_strength'], cfg['end_strength'], cfg['perturbation_number'])
        for i in tqdm(range(cfg["perturbation_number"])):
            per_dataset = dataset_perturbation_function(dataset, strength=strength[i])
            per_loss = model_eval_function(model, per_dataset, cfg)
            per_losses.append(np.expand_dims(per_loss, -1))
            ref_per_loss = model_eval_function(self.reference_model, per_dataset, cfg) if cfg["calibration"] else None
            try:
                ref_per_losses.append(np.expand_dims(ref_per_loss, -1))
            except:
                pass
        per_losses = np.concatenate(per_losses, axis=-1)
        var_losses = per_losses - np.expand_dims(ori_losses, -1)
        ref_per_losses = np.concatenate(ref_per_losses, axis=-1) if cfg["calibration"] else None
        ref_var_losses = ref_per_losses - np.expand_dims(ref_ori_losses, -1) if cfg["calibration"] else None

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
        return output

    def gen_data_diffusion(self, model, img_path, sample_numbers=100, batch_size=100, path="attack/attack_data_diffusion"):
        pipeline = model
        generated_samples = []
        for i in range(0, sample_numbers, batch_size):
            gen = pipeline(batch_size).images
            generated_samples.extend(gen)
        utils.create_folder(img_path)
        for i, img in enumerate(generated_samples):
            img.save(img_path + f"/{i}.jpg")
        files = utils.get_file_names(img_path)
        dataset = Dataset.from_dict({"image": files}).cast_column("image", Image())
        return dataset

    def data_prepare(self, kind, cfg):
        logger.info("Preparing data...")
        data_path = os.path.join(PATH, cfg["attack_data_path"], f"attack_data_{cfg['target_model']}@{cfg['dataset']}")
        target_model = getattr(self, kind + "_model")
        mem_data = self.datasets[kind]["train"]
        nonmem_data = self.datasets[kind]["valid"]

        mem_path = os.path.join(data_path, kind, "mem_feat.npz")
        nonmem_path = os.path.join(data_path, kind, "nonmen_feat.npz")
        ref_mem_path = os.path.join(data_path, kind, "ref_mem_feat.npz")
        ref_nonmem_path = os.path.join(data_path, kind, "ref_nonmen_feat.npz")

        pathlist = (mem_path, nonmem_path, ref_mem_path, ref_nonmem_path) if cfg["calibration"] else (mem_path, nonmem_path)

        if not utils.check_files_exist(*pathlist) or not cfg["load_attack_data"]:

            logger.info("Generating feature vectors for memory data...")
            mem_feat, ref_mem_feat = self.eval_perturb(target_model, mem_data, cfg)
            utils.save_dict_to_npz(mem_feat, mem_path)
            if cfg["calibration"]:
                utils.save_dict_to_npz(ref_mem_feat, ref_mem_path)

            logger.info("Generating feature vectors for non-memory data...")
            nonmem_feat, ref_nonmem_feat = self.eval_perturb(target_model, nonmem_data, cfg)
            utils.save_dict_to_npz(nonmem_feat, nonmem_path)
            if cfg["calibration"]:
                utils.save_dict_to_npz(ref_nonmem_feat, ref_nonmem_path)

            logger.info("Saving feature vectors...")

        else:
            logger.info("Loading feature vectors...")
            mem_feat = utils.load_dict_from_npz(mem_path)
            ref_mem_feat = utils.load_dict_from_npz(ref_mem_path) if cfg["calibration"] else None
            nonmem_feat = utils.load_dict_from_npz(nonmem_path)
            ref_nonmem_feat = utils.load_dict_from_npz(ref_nonmem_path) if cfg["calibration"] else None

        logger.info("Data preparation complete.")

        return Dict(
            mem_feat=mem_feat,
            nonmem_feat=nonmem_feat,
            ref_mem_feat=ref_mem_feat,
            ref_nonmem_feat=ref_nonmem_feat,
                    )


    def feat_prepare(self, info_dict, cfg):
        # mem_info = info_dict.mem_feat
        # ref_mem_info = info_dict.ref_mem_feat
        if cfg["calibration"]:
            mem_feat = info_dict.mem_feat.var_losses / np.expand_dims(info_dict.mem_feat.ori_losses, -1)\
                       - info_dict.ref_mem_feat.ref_var_losses / np.expand_dims(info_dict.ref_mem_feat.ref_ori_losses, -1)
            nonmem_feat = info_dict.nonmem_feat.var_losses / np.expand_dims(info_dict.nonmem_feat.ori_losses, -1)\
                       - info_dict.ref_nonmem_feat.ref_var_losses / np.expand_dims(info_dict.ref_nonmem_feat.ref_ori_losses, -1)
            # gen_feat = info_dict.gen_feat.var_losses / info_dict.gen_feat.ori_losses[:, :, None] \
            #               - info_dict.ref_gen_feat.ref_var_losses / info_dict.ref_gen_feat.ref_ori_losses[:, :, None]
        else:
            mem_feat = info_dict.mem_feat.var_losses / np.expand_dims(info_dict.mem_feat.ori_losses, -1)
            nonmem_feat = info_dict.nonmem_feat.var_losses / np.expand_dims(info_dict.nonmem_feat.ori_losses, -1)
            # gen_feat = info_dict.gen_feat.var_losses / info_dict.gen_feat.ori_losses[:, :, None]
        # if cfg["target_model"] == "diffusion":
        #     mem_feat = mem_feat[:, 2, :]
        #     nonmem_feat = nonmem_feat[:, 2, :]
            # gen_feat = gen_feat[:, 2, :]

        if cfg["attack_kind"] == "stat":
            mem_feat = mem_feat[:, :, 5]
            nonmem_feat = nonmem_feat[:, :, 5]
            mem_feat[np.isnan(mem_feat)] = 0
            nonmem_feat[np.isnan(nonmem_feat)] = 0
            feat = np.concatenate([mem_feat.mean(axis=(-1)), nonmem_feat.mean(axis=(-1))])
            ground_truth = np.concatenate([np.zeros(mem_feat.shape[0]), np.ones(nonmem_feat.shape[0])]).astype(np.int)

        elif cfg["attack_kind"] == "nn":
            # mem_freq = self.frequency(mem_feat, split=100)
            # nonmem_freq = self.frequency(nonmem_feat, split=100)
            # mem_feat, nonmem_feat = utils.ndarray_to_tensor(mem_freq, nonmem_freq)
            mem_feat, nonmem_feat = utils.ndarray_to_tensor(mem_feat, nonmem_feat)
            if cfg["target_model"] == "vae":
                mem_feat.sort(axis=1)
                nonmem_feat.sort(axis=1)
            feat = torch.cat([mem_feat, nonmem_feat])
            feat[torch.isnan(feat)] = 0
            # if cfg["target_model"] == "diffusion":
            feat = feat.unsqueeze(1)
            ground_truth = torch.cat([torch.zeros(mem_feat.shape[0]), torch.ones(nonmem_feat.shape[0])]).type(torch.LongTensor).cuda()
        return feat, ground_truth

    def attack_model_training(self, cfg):

        save_path = os.path.join(PATH, cfg["attack_data_path"], f"attack_model_{cfg['target_model']}@{cfg['dataset']}", 'attack_model.pth')

        raw_info = self.data_prepare("shadow", cfg)
        eval_raw_info = self.data_prepare("target", cfg)

        feat, ground_truth = self.feat_prepare(raw_info, cfg)
        eval_feat, eval_ground_truth = self.feat_prepare(eval_raw_info, cfg)

        feature_dim = feat.shape[-1]
        # attack_model = MLAttckerModel(feature_dim, output_size=2).cuda()
        # attack_model = ResNet18(num_channels=1, num_classes=2).cuda() if cfg["target_model"] == "diffusion" \
        #     else MLAttckerModel(feature_dim, output_size=2).cuda()
        attack_model = ResNet18(num_channels=1, num_classes=2).cuda()
        if cfg["load_trained"] and utils.check_files_exist(save_path):
            attack_model.load_state_dict(torch.load(save_path))
            self.attack_model = attack_model
            self.is_model_training = True
            return
        optimizer = optim.Adam(attack_model.parameters(), lr=0.01, weight_decay=0)
        # schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[])
        weight = torch.Tensor([1, 1]).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        print_freq = 1
        for i in range(cfg["epoch_number"]):
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
                # ground_truth, predict, eval_ground_truth, eval_predict = utils.tensor_to_ndarray(ground_truth, predict, eval_ground_truth, eval_predict)
                self.eval_attack(ground_truth, predict[:, 1], plot=False)
                self.eval_attack(eval_ground_truth, eval_predict[:, 1], plot=False)
        self.is_model_training = True
        self.attack_model = attack_model
        # Save model
        folder_path = os.path.dirname(save_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(attack_model.state_dict(), save_path)

    def conduct_attack(self, cfg):
        if cfg["attack_kind"] == 'nn':
            if not self.is_model_training:
                self.attack_model_training(cfg)
            attack_model = self.attack_model
            raw_info = self.data_prepare("target", cfg)
            feat, ground_truth = self.feat_prepare(raw_info, cfg)
            predict = attack_model(feat)
            # predict, ground_truth = utils.tensor_to_ndarray(predict, ground_truth)
            self.eval_attack(ground_truth, predict[:, 1])
        elif cfg["attack_kind"] == 'stat':
            raw_info = self.data_prepare("target", cfg)
            feat, ground_truth = self.feat_prepare(raw_info, cfg)
            # self.distinguishability_plot(raw_info['mem_feat']['ori_losses'].mean(-1),
            #                              raw_info['nonmem_feat']['ori_losses'].mean(-1))
            # self.distinguishability_plot(feat[:1000], feat[-1000:])
            self.eval_attack(ground_truth, -feat)

    def attack_demo(self, cfg, pipeline, timestep=200):
        mem_data = self.datasets["target"]["train"]
        nonmem_data = self.datasets["target"]["valid"]
        mem_data.set_transform(utils.transform_images)
        nonmem_data.set_transform(utils.transform_images)
        ddpm_score = []
        ddim_score = []
        for dataset in [mem_data, nonmem_data]:
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=cfg["eval_batch_size"],
                shuffle=False,
                num_workers=32,
                pin_memory=torch.cuda.is_available()
            )
            ddpm_result = []
            ddim_result = []
            for iteration, batch in tqdm(enumerate(data_loader)):
                clean_images = batch["input"].cuda()
                # start_time = time.time()
                ddpm_loss = self.ddpm_loss(pipeline, clean_images, timestep)
                ddim_loss = self.ddim_loss(pipeline, clean_images, t_sec=timestep)
                ddpm_result.append(ddpm_loss)
                ddim_result.append(ddim_loss)
            ddpm_result = np.concatenate(ddpm_result, axis=0)
            ddim_result = np.concatenate(ddim_result, axis=0)
            ddpm_score.append(ddpm_result)
            ddim_score.append(ddim_result)
        label = np.concatenate([np.zeros(mem_data.shape[0]), np.ones(nonmem_data.shape[0])]).astype(np.int)
        # label = torch.cat([torch.zeros(mem_data.shape[0]), torch.ones(nonmem_data.shape[0])]).type(torch.LongTensor).cuda()
        ddim_score = np.concatenate(ddim_score, axis=0)
        ddpm_score = np.concatenate(ddpm_score, axis=0)
        self.eval_attack(label, ddim_score)  # function eval_attack needs to be revised.
        self.eval_attack(label, ddpm_score)

    # @staticmethod
    # def eval_attack(y_true, y_scores, plot=True):
    #     n_classes = 3
    #     y_true, y_scores = utils.tensor_to_ndarray(y_true, y_scores)
    #     y_true = utils.convert_labels_to_one_hot(y_true, num_classes=n_classes)
    #     fpr = dict()
    #     tpr = dict()
    #     roc_auc = dict()
    #     for i in range(n_classes):
    #         fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
    #         roc_auc[i] = auc(fpr[i], tpr[i])
    #     # Compute micro-average AUC
    #     fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_scores.ravel())
    #     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #     logger.info(f"Auc on the target model: {roc_auc['micro']}")
    #     class_name = ["Member", "Non-Member", "Generative"]
    #     if plot:
    #         # Plot AUC curves for each class
    #         plt.figure()
    #         lw = 2
    #         colors = cycle(['red', 'darkorange', 'green'])
    #         for i, color in zip(range(n_classes), colors):
    #             plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #                      label='ROC curve on {0} data (AUC = {1:0.2f})'.format(class_name[i], roc_auc[i]))
    #
    #         # Plot the micro-average ROC curve
    #         plt.plot(fpr["micro"], tpr["micro"], color='green', linestyle=':', linewidth=4,
    #                  label='Overall ROC curve (AUC = {0:0.2f})'
    #                        ''.format(roc_auc["micro"]))
    #
    #         # Plot the randomized ROC curve
    #         plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    #         plt.xlim([0.0, 1.0])
    #         plt.ylim([0.0, 1.05])
    #         plt.xlabel('False Positive Rate')
    #         plt.ylabel('True Positive Rate')
    #         plt.title('Discriminative Performance on Different Categories of Data.')
    #         plt.legend(loc="lower right")
    #         plt.show()
    @staticmethod
    def heatmap_plot(data, **kwargs):
        ax = sns.heatmap(data, annot=True,
                         # cmap='crest',
                         fmt='.2f', **kwargs)
        ax.set_ylabel(r'Perturbation Strength Factor $\lambda$', fontsize=18)
        ax.set_xlabel('Time step $t$', fontsize=18)
        ax.set_yticklabels(['{:.2f}'.format(label) for label in np.linspace(0.95, 0.7, 10)],
                           fontsize=14, rotation=45)
        ax.set_xticklabels([0, 50, 100, 150, 200, 250, 300, 350, 400, 450], fontsize=14, rotation=45)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        plt.tight_layout()
        plt.savefig("heat-diffusion-naive.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    @staticmethod
    def vec_heatmap(vec, **kwargs):
        vec = vec.reshape((1, -1))
        ax = sns.heatmap(vec, annot=True,
                         yticklabels=False,
                         square=True,
                         cbar=False,
                         annot_kws={"fontsize": 14},
                         # cmap='crest',
                         fmt='.2f', **kwargs)
        ax.set_xticklabels([0, 50, 100, 150, 200, 250, 300, 350, 400, 450], fontsize=14)
        ax.set_xlabel('Time step $t$', fontsize=18)
        plt.tight_layout()
        plt.savefig("heat-diffusion-naive.pdf", format="pdf", bbox_inches="tight")
        plt.show()
    @staticmethod
    def distinguishability_plot(mem, non_mem):
        sns.set_theme()
        mem_color = "indianred"
        non_mem_color = "forestgreen"
        sns.kdeplot(mem, fill=True, color=mem_color, alpha=0.5)
        sns.kdeplot(non_mem, fill=True, color=non_mem_color, alpha=0.5)

        mem_mean = round(mem.mean(), 2)
        mem_std = round(mem.std(), 2)
        non_mem_mean = round(non_mem.mean(), 2)
        non_mem_std = round(non_mem.std(), 2)

        # plt.xlabel(r"${\mathcal{F}}({x}, \theta)$", fontsize=22, labelpad=10)
        plt.xlabel(r"$L_{\rm{ELOB}}\left({x}\right)$", fontsize=22, labelpad=10)
        plt.ylabel('Density', fontsize=22, labelpad=10)
        plt.legend(['Member', 'Non-member'], fontsize=20, loc='upper right')
        plt.xlim([0.02, 0.09])
        mem_text = '\n'.join((
                    r'$\mu_{Mem}=%.2f$' % (mem_mean, ),
                    r'$\sigma_{Mem}=%.2f$' % (mem_std, )))
        non_mem_text = '\n'.join((
                    r'$\mu_{Non}=%.2f$' % (non_mem_mean, ),
                    r'$\sigma_{Non}=%.2f$' % (non_mem_std, )))
        mem_props = dict(boxstyle='round', facecolor=mem_color, alpha=0.15, edgecolor='black')
        non_mem_props = dict(boxstyle='round', facecolor=non_mem_color, alpha=0.15, edgecolor='black')

        plt.tick_params(labelsize=16)
        plt.text(0.04, 0.6, mem_text, transform=plt.gca().transAxes, fontsize=22, bbox=mem_props)
        plt.text(0.63, 0.25, non_mem_text, transform=plt.gca().transAxes, fontsize=22, bbox=non_mem_props)

        plt.tight_layout()
        plt.savefig("distinguishability-diffusion-naive.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    @staticmethod
    def eval_attack(y_true, y_scores, plot=True):
        if type(y_true) == torch.Tensor:
            y_true, y_scores = utils.tensor_to_ndarray(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)
        logger.info(f"AUC on the target model: {auc_score}")

        # Finding the threshold point where FPR + TPR equals 1
        threshold_point = tpr[np.argmin(np.abs(tpr - (1 - fpr)))]
        logger.info(f"ASR on the target model: {threshold_point}")

        # Finding the threshold point where FPR + TPR equals 1
        tpr_1fpr = tpr[np.argmin(np.abs(fpr - 0.01))]
        logger.info(f"TPR@1%FPR on the target model: {tpr_1fpr}")


        if plot:
            # plot the ROC curve
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score}; ASR = {threshold_point})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            # plot the no-skill line for reference
            plt.plot([0, 1], [0, 1], linestyle='--')
            # show the plot
            plt.show()
    @staticmethod
    def sentence_perturb(dataset, embedding, rate=0.1): #  TODO: whether the perturb number should be identical for sentences with different length?
        per_dataset = deepcopy(dataset)
        sim = torch.mm(embedding, embedding.T)
        prop = F.softmax(sim.fill_diagonal_(float('-inf')), dim=1).cpu().numpy()
        for idx in range(len(per_dataset)):
            ori_data = per_dataset[idx]
            sen_len = ori_data["length"]
            ori_sen = ori_data["input"][1:sen_len]
            per_sen = []
            for word in ori_sen:
                assert word not in [0, 2, 3]
                if random.random() < rate:
                    per_word = int(np.random.choice(len(prop[word, :]), p=prop[word, :]))
                    per_sen.append(per_word)
                else:
                    per_sen.append(word)
            input_sen = [2] + per_sen
            input_sen.extend([0]*(60-sen_len))
            target_sen = per_sen + [3]
            target_sen.extend([0]*(60-sen_len))
            per_data = {
                "input": input_sen,
                "target": target_sen,
                "length": sen_len
            }
            per_dataset.data[str(idx)] = per_data
        return per_dataset


    @staticmethod
    def gaussian_noise_tensor(tensor, mean=0.0, std=0.1):
        # create a tensor of gaussian noise with the same shape as the input tensor
        noise = torch.randn(tensor.shape) * std + mean
        # add the noise to the original tensor
        noisy_tensor = tensor + noise
        # make sure the pixel values are within [0, 1]
        noisy_tensor = torch.clamp(noisy_tensor, -1.0, 1.0)
        return noisy_tensor

    @staticmethod
    def norm_transform_images(examples):
        # Preprocessing the datasets and DataLoaders creation.
        # if norm: -1 : 1
        augmentations = transforms.Compose(
            [
                transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(64) if True else transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip() if False else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]
        )
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    @staticmethod
    def transform_images(examples):
        # Preprocessing the datasets and DataLoaders creation.
        # if norm: -1 : 1
        augmentations = transforms.Compose(
            [
                transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(64) if True else transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip() if False else transforms.Lambda(lambda x: x),
                transforms.ToTensor()
            ]
        )
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    @staticmethod
    def norm_image_dataset_perturbation(dataset, strength):
        perturbation = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 0.8)),
            # transforms.CenterCrop(size=int(64 * strength)),
            # transforms.Resize(size=64),
            # transforms.RandomPerspective(distortion_scale=strength, p=1),
            transforms.Resize(size=int(strength)),
            transforms.Resize(size=64),
            # transforms.ColorJitter(hue=(strength, strength)),
            transforms.Normalize([0.5], [0.5]),
        ])
        def transform_images(examples):
            images = [perturbation(image.convert("RGB")) for image in examples["image"]]
            return {"input": images}
        per_dataset = deepcopy(dataset)
        per_dataset.set_transform(transform_images)

        return per_dataset

    @staticmethod
    def image_dataset_perturbation(dataset, strength):
        perturbation = transforms.Compose([
            transforms.ToTensor(),
            # transforms.CenterCrop(size=int(64 * strength)),
            # transforms.Resize(size=64),
            # transforms.ColorJitter(brightness=(strength, strength)),
            transforms.RandomRotation(degrees=(strength, strength)),
        ])
        def transform_images(examples):
            images = [perturbation(image.convert("RGB")) for image in examples["image"]]
            return {"input": images}
        per_dataset = deepcopy(dataset)
        per_dataset.set_transform(transform_images)

        return per_dataset

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