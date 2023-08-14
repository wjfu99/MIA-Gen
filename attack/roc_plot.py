import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Load abs path
PATH = os.path.dirname(os.path.abspath(__file__))
# attack_data_path = "attack"
target_model = "diffusion"
dataset = "celeba"
data_path = os.path.join(PATH, f"attack_data_{target_model}@{dataset}", 'roc.npz')
roc = np.load(data_path)
fpr = roc['fpr']
tpr = roc['tpr']
closest_point_index = np.argmax(tpr - fpr)
closest_point = (fpr[closest_point_index], tpr[closest_point_index])

roc_list = [{
    "data_path": os.path.join(PATH, f"attack_data_{target_model}@{dataset}", 'roc_nn.npz'),
    "legend": r"\text{PFAMI}_{\textit{NNs}}"
}, {
    "data_path": os.path.join(PATH, f"attack_data_{target_model}@{dataset}", 'roc_stat.npz'),
    "legend": r"\text{PFAMI}_{\textit{Met}}"
}]


def add_roc(ax, roc):
    data = np.load(roc["data_path"])
    fpr = data['fpr']
    tpr = data['tpr']
    data = np.array([fpr, tpr])
    sns.lineplot(data, x=0, y=1, ax=ax)


# Set seaborn plot style
sns.set(style='whitegrid')
# Plot ROC curve
fig, ax = plt.subplots(figsize=(8, 6))
for roc in roc_list:
    add_roc(ax, roc)
# sns.lineplot([[0, 1], [0, 1]], color='r', linestyle='--', ax=ax)
# sns.scatterplot([[closest_point[0]], [closest_point[1]]], color='b', s=100)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(labels=['legendEntry1', 'legendEntry2'])
# plt.legend(loc='lower right')
plt.show()