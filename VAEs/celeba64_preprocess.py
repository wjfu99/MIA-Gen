import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

def imgs_to_npz():
    npz = []

    for img in tqdm(os.listdir("./data/celeba64/training")):
        img_arr = cv2.imread("./data/celeba64/training/" + img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)  # bgr default in cv2
        npz.append(img_arr)
    for img in os.listdir("./data/celeba64/testing"):
        img_arr = cv2.imread("./data/celeba64/testing/" + img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)  # bgr default in cv2
        npz.append(img_arr)
    for img in os.listdir("./data/celeba64/validation"):
        img_arr = cv2.imread("./data/celeba64/validation/" + img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)  # bgr default in cv2
        npz.append(img_arr)
    output_npz = np.array(npz)
    output_npz = np.transpose(output_npz, (0, 3, 1, 2))
    np.savez('./data/celeba64/celeba64.npz', output_npz)
    print(f"{output_npz.shape} size array saved into celeba64_train.npz")  # (202599, 64, 64, 3)

imgs_to_npz()