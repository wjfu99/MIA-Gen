import os
import numpy as np
import torch
import pythae
import torch.nn.functional as F
from imageio import imwrite
from pythae.models import AutoModel
from pythae.samplers import NormalSampler
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"

last_training = sorted(os.listdir('my_model'))[-1]
trained_model = AutoModel.load_from_folder(os.path.join('my_model', last_training, 'final_model'))
trained_model = trained_model.to(device)

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

mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)

train_dataset = mnist_trainset.data[:-10000].reshape(-1, 1, 28, 28) / 255.
eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.


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


def eval_loss(input):
    input = {"data": input.to(device)}
    output = trained_model(input)
    recon_x = output.recon_x
    x = input['data']
    recon_loss = F.mse_loss(
        recon_x.reshape(x.shape[0], -1),
        x.reshape(x.shape[0], -1),
        reduction="none",
    ).sum(dim=-1)
    return recon_loss.item()

def mask_tensor(tensor, prob, num_masks=1):
    masks = []

    for i in range(num_masks):
        # create a tensor of binary values with the same shape as the input tensor
        mask = torch.bernoulli(torch.full(tensor.shape, prob))
        # apply the mask on the original tensor
        masked_tensor = tensor * mask
        masks.append(masked_tensor)
    return masks

def add_gaussian_noise(tensor, noise_scale, num_noised=1):
    noised_tensors = []
    for i in range(num_noised):
        # generate random noise tensor
        noise = torch.randn(tensor.size()) * noise_scale
        # add noise to the input tensor
        noised_tensor = tensor + noise
        noised_tensors.append(noised_tensor)
    return noised_tensors

def eval_perturb(dataset):
    losses = []
    ori_losses = []
    var_losses = []
    per_losses = []
    per_num = 100
    for data in dataset:
        ori_loss = eval_loss(data)
        # masks = mask_tensor(data, prob=0.3, num_masks=10)
        masks = add_gaussian_noise(data, noise_scale=0.1, num_noised=per_num)
        per_loss = []
        avg_loss = 0
        for mask in masks:
            recon_loss = eval_loss(mask)
            per_loss.append(recon_loss)
            avg_loss += recon_loss
        avg_loss = avg_loss / per_num
        ori_losses.append(ori_loss)
        per_losses.append(per_loss)
        losses.append(avg_loss)
        var_losses.append(avg_loss-ori_loss)
    output = {
        'losses': np.array(losses),
        'var_losses': np.array(var_losses),
        'per_losses': np.array(per_losses),
        'ori_losses': np.array(ori_losses)
    }
    return output


eval_losses = eval_perturb(eval_dataset[:25])
train_losses = eval_perturb(train_dataset[:25])

plt_num = 10
for i in range(plt_num):
    train = train_losses['per_losses'][i] - train_losses['ori_losses'][i]
    eval = eval_losses['per_losses'][i] - eval_losses['ori_losses'][i]
    sns.kdeplot(train, fill=True, color='red')
    sns.kdeplot(eval,  fill=True, color='blue')

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