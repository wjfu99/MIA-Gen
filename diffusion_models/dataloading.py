# from diffusers import DiffusionPipeline
# import torch
# model_id = "./ddpm-ema-flowers-64"
# pipeline = DiffusionPipeline.from_pretrained(model_id)
#
# prompt = "portrait photo of a old warrior chief"
#
# pipeline = pipeline.to("cuda")
#
# generator = torch.Generator("cuda").manual_seed(0)
# image = pipeline(generator=generator, num_inference_steps=1000).images[0]
#
# image.save("ddpm_generated_image.png")
from datasets import load_dataset
import time
import os
from datasets import Dataset, Image

def get_file_names(folder_path):
    # List to store the file names
    file_names = []

    # Loop through each file in the folder
    for file_name in sorted(os.listdir(folder_path)):
        # Check if the current item is a file
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(os.path.join(folder_path, file_name))

    return file_names

start_time = time.time()
files = get_file_names("/mnt/data0/fuwenjie/MIA/MIA-Gen/VAEs/data/celeba64/total")

# files = sorted(files)
# files = "/mnt/data0/fuwenjie/MIA/MIA-Gen/VAEs/data/celeba64/total/000001.jgp"
dataset = Dataset.from_dict({"image": files}).cast_column("image", Image())
# dataset = load_dataset("huggan/flowers-102-categories", split="train")
end_time = time.time()
times = start_time - end_time

flower_dataset = load_dataset("huggan/flowers-102-categories", split="train")
#
# dataset = Dataset.from_dict({"image": ["/mnt/data0/fuwenjie/MIA/MIA-Gen/VAEs/data/celeba64/training",
#                                        "/mnt/data0/fuwenjie/MIA/MIA-Gen/VAEs/data/celeba64/testing",
#                                        "/mnt/data0/fuwenjie/MIA/MIA-Gen/VAEs/data/celeba64/validation"]}).cast_column("image", Image())
