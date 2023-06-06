import os
import numpy as np
import torch
import pythae
from imageio import imwrite
from pythae.models import AutoModel
from pythae.samplers import NormalSampler
import torchvision.datasets as datasets

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

input = {"data": eval_dataset[:25].to(device)}
## Reconstructions
reconstructions = trained_model(input).recon_x.detach().cpu()
output_dir='./data/mnist/gen_data'
for j in range(len(reconstructions)):
    save_img(
        reconstructions[j], output_dir, "%08d.png" % int(j)
    )

output_dir='./data/mnist/eval_data'
for j in range(len(eval_dataset[:25])):
    save_img(
        eval_dataset[:25][j], output_dir, "%08d.png" % int(j)
    )