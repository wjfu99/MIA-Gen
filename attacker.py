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

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"



# # create normal sampler
# normal_samper = NormalSampler(
#     model=trained_model
# )
#
# # sample
# gen_data = normal_samper.sample(
#     num_samples=25,
#     output_dir='./data/mnist/gen_data'
# )

dataset = "celeba"

target_model = sorted(os.listdir(PATH + '/target_model/my_models_on_'+dataset))[-2]
trained_model = AutoModel.load_from_folder(os.path.join(PATH + '/target_model/my_models_on_'+dataset, target_model, 'final_model'))
trained_model = trained_model.to(device)

reference_model = sorted(os.listdir(PATH + '/target_model/my_models_on_'+dataset))[-1]
reference_model = AutoModel.load_from_folder(os.path.join(PATH + '/target_model/my_models_on_'+dataset, reference_model, 'final_model'))
reference_model = reference_model.to(device)

# last_training = sorted(os.listdir('target_model/my_model'))[-1]
# trained_model = AutoModel.load_from_folder(os.path.join('target_model/my_model', last_training, 'final_model'))
# trained_model = trained_model.to(device)
if dataset == "celeba":
    celeba64_dataset = np.load("./target_model/data/celeba64/celeba64.npz")["arr_0"] / 255.0
    train_data = torch.Tensor(celeba64_dataset[:10000])
    eval_data = torch.Tensor(celeba64_dataset[10000:13000])
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

# mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)
#
# train_data = mnist_trainset.data[:-10000].reshape(-1, 1, 28, 28) / 255.
# eval_data = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.

def save_img(img_tensor: torch.Tensor, dir_path: str, img_name: str):
    """Saves a data point as .png file in dir_path with img_name as name.

    Args:
        img_tensor (torch.Tensor): The image of shape CxHxW in the range [0-1]
        dir_path (str): The folder where in which the images must be saved
        ig_name (str): The name to apply to the file containing the image.
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"--> Created folder {dir_path}. Images will be saved here")

    img = 255.0 * torch.movedim(img_tensor, 0, 2).cpu().detach().numpy()
    if img.shape[-1] == 1:
        img = np.repeat(img, repeats=3, axis=-1)

    img = img.astype("uint8")
    imwrite(os.path.join(dir_path, f"{img_name}"), img)

def show_image(image):
    # show reconstructions
    if image.shape[0] == 1:
        fig, axes = plt.subplots(figsize=(10, 10))
        axes.imshow(image.cpu().squeeze(), cmap='gray')
        axes.axis('off')
        plt.tight_layout(pad=0.)
        plt.show()
    elif image.shape[0] == 3:
        pil_image = ToPILImage()(image.cpu())
        fig, axes = plt.subplots(figsize=(10, 10))
        axes.imshow(pil_image)
        axes.axis('off')
        plt.tight_layout(pad=0.)
        plt.show()

def eval_loss(input, refer=False):
    input = {"data": input.to(device)}
    if not refer:
        output = trained_model(input)
    else:
        output = reference_model(input)
    recon_loss = output.recon_loss
    reg_loss = output.reg_loss
    loss = output.loss
    return output

def mask_tensor(tensor, prob, num_masks=1):
    masks = []

    for i in range(num_masks):
        # create a tensor of binary values with the same shape as the input tensor
        mask = torch.bernoulli(torch.full(tensor.shape[-2:], 1-prob))
        # apply the mask on the original tensor
        masked_tensor = tensor * mask
        masks.append(masked_tensor)
    masks = torch.stack(masks, dim=0)
    masks = torch.squeeze(masks)
    return masks

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

def add_gaussian_noise(tensor, noise_scale, num_noised=1):
    noised_tensors = []
    for i in range(num_noised):
        # generate random noise tensor
        noise = torch.randn(tensor.size()) * noise_scale
        # add noise to the input tensor
        noised_tensor = tensor + noise
        noised_tensors.append(noised_tensor)
    return noised_tensors

def eval_perturb(dataset, refer=False):
    """
    Evaluate the loss of the perturbed data

    :param dataset: N*1*28*28
    :return: losses: N*1; var_losses: N*1; per_losses: N*Mask_Num; ori_losses: N*1
    """
    losses = []
    ori_losses = []
    var_losses = []
    per_losses = []
    peak_losses = []
    per_num = 100
    for data in tqdm(dataset):
        data = torch.unsqueeze(data, 0)
        # show_image(data[0])
        # show_image(eval_loss(data).recon_x.detach()[0])
        ori_loss = eval_loss(data, refer).loss.item()
        # masks = mask_tensor(data, prob=0.05, num_masks=per_num)
        masks = gaussian_noise_tensor(data, 0, 0.1, per_num)
        # masks = add_gaussian_noise(data, noise_scale=0.1, num_noised=per_num)
        per_loss = []
        avg_loss = 0
        for mask in masks:
            mask = torch.unsqueeze(mask, 0)
            recon_loss = eval_loss(mask, refer).loss.item()
            per_loss.append(recon_loss)
            avg_loss += recon_loss
        avg_loss = avg_loss / per_num
        ori_losses.append(ori_loss)
        per_losses.append(per_loss)
        losses.append(avg_loss)
        var_losses.append(avg_loss-ori_loss)
        peak_losses.append(np.min(per_loss) - ori_loss)
    output = {
        'per_losses': np.array(per_losses),
        'ori_losses': np.array(ori_losses),
        'losses': np.array(losses),
        'var_losses': np.array(var_losses),
        'peak_losses': np.array(peak_losses)
    }
    return output


eval_losses = eval_perturb(eval_data[:1000])
train_losses = eval_perturb(train_data[5000:6000])

ref_eval_losses = eval_perturb(eval_data[:1000], True)
ref_train_losses = eval_perturb(train_data[5000:6000], True)

plt_num = 5
for i in range(plt_num):
    train = train_losses['per_losses'][i] - train_losses['ori_losses'][i]
    eval = eval_losses['per_losses'][i] - eval_losses['ori_losses'][i]
    sns.kdeplot(train, fill=True, color='red', alpha=0.5)
    sns.kdeplot(eval, fill=True, color='blue', alpha=0.5)
plt.xlabel('Increase of loss')
plt.ylabel('Density')
plt.legend(['Member', 'Non-member'])  # Add a single legend with both labels
plt.show()

# train = np.min(train_losses['per_losses'] - train_losses['ori_losses'], axis=1)
# eval = np.min(eval_losses['per_losses'] - eval_losses['ori_losses'], axis=1)
# train = np.min(train_losses['per_losses'] - train_losses['ori_losses'][:, None], axis=1)
# eval = np.min(eval_losses['per_losses'] - eval_losses['ori_losses'][:, None], axis=1)
train = np.min(train_losses['per_losses'] - train_losses['ori_losses'][:, None], axis=1)/train_losses['ori_losses']
eval = np.min(eval_losses['per_losses'] - eval_losses['ori_losses'][:, None], axis=1)/eval_losses['ori_losses']

ref_train = np.min(ref_train_losses['per_losses'] - ref_train_losses['ori_losses'][:, None], axis=1)/ref_train_losses['ori_losses']
ref_eval = np.min(ref_eval_losses['per_losses'] - ref_eval_losses['ori_losses'][:, None], axis=1)/ref_eval_losses['ori_losses']

sns.kdeplot(train-ref_train, fill=True, color='red', alpha=0.5)
sns.kdeplot(eval-ref_eval, fill=True, color='blue', alpha=0.5)
plt.xlabel('Minimum increase of loss')
plt.ylabel('Density')
plt.legend(['Member', 'Non-member'])  # Add a single legend with both labels
plt.show()

# evaluate the attack performance
y_scores = np.concatenate([train, eval], axis=0)
y_true = np.concatenate([np.ones((100,)), np.zeros(100,)], axis=0)
auc_score = roc_auc_score(y_true, y_scores)

# calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc_score)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

# plot the no-skill line for reference
plt.plot([0, 1], [0, 1], linestyle='--')

# show the plot
plt.show()

# calculate the precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# calculate the F1-score
fscore = (2 * precision * recall) / (precision + recall + 10e-6)  # calculate the f1 score
f1 = fscore.max()

# plot the PR curve
plt.plot(recall, precision, label='PR curve (F1-score = %0.2f)' % f1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

# show the plot
plt.show()
# input = {"data": eval_dataset[:25].to(device)}
# ## Reconstructions
# reconstructions = trained_model(input).recon_x.detach().cpu()
# output_dir='./data/mnist/gen_data'
# for j in range(len(reconstructions)):
#     save_img(
#         reconstructions[j], output_dir, "%08d.png" % int(j)
#     )
#
# output_dir='./data/mnist/eval_data'
# for j in range(len(eval_dataset[:25])):
#     save_img(
#         eval_dataset[:25][j], output_dir, "%08d.png" % int(j)
#     )