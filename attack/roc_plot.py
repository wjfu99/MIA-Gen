import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load abs path
PATH = os.path.dirname(os.path.abspath(__file__))
# attack_data_path = "attack"
target_model = "diffusion"
dataset = "celeba"
def load_roc(data_path):
    roc = np.load(data_path)
    roc = pd.DataFrame(roc['tpr'], index=roc['fpr'])
    roc = roc[~roc.index.duplicated(keep='first')]
    roc = roc[0]
    # roc = (roc.reset_index()
    # .drop_duplicates(subset='index', keep='last')
    # .set_index('index').sort_index())
    return roc
# closest_point_index = np.argmax(tpr - fpr)
# closest_point = (fpr[closest_point_index], tpr[closest_point_index])
pfami_nn = load_roc(os.path.join(PATH, f"attack_data_{target_model}@{dataset}", 'roc_nn.npz'))
pfami_met = load_roc(os.path.join(PATH, f"attack_data_{target_model}@{dataset}", 'roc_stat.npz'))
secmi_nn = load_roc("/mnt/data0/fuwenjie/MIA/SecMI/mia_evals/roc_nns.npz")
secmi_met = load_roc("/mnt/data0/fuwenjie/MIA/SecMI/mia_evals/roc_stat.npz")



roc_df = pd.DataFrame({
    r'${\rm PFAMI}_{NNs}$': pfami_nn,
    r'${\rm PFAMI}_{Met}$': pfami_met,
    r'${\rm SecMI}_{NNs}$': secmi_nn,
    r'${\rm SecMI}_{Stat}$': secmi_met,
})


# def add_roc(roc):
#     data = np.load(roc["data_path"])
#     fpr = data['fpr']
#     tpr = data['tpr']
#     data = np.array([fpr, tpr])
#     # plt.plot(fpr, tpr)
#     sns.lineplot([[0, 1], [0, 1]], linestyle='--')


# Set seaborn plot style
sns.set_theme()
# Plot ROC curve
plt.subplots(figsize=(8, 6))
# for roc in roc_list:
#     add_roc(roc)
sns.lineplot(roc_df, linewidth=5)
# sns.lineplot([[0, 1], [0, 1]], color='r', linestyle='--', ax=ax)
# sns.scatterplot([[closest_point[0]], [closest_point[1]]], color='b', s=100)
plt.xlabel("False Positive Rate", fontsize=24, labelpad=10)
plt.ylabel('True Positive Rate', fontsize=24, labelpad=10)
plt.tick_params(labelsize=20)
leg = plt.legend(fontsize=26, loc='lower right')
for legobj in leg.legendHandles:
    legobj.set_linewidth(5)
plt.savefig(os.path.join(PATH, f"attack_data_{target_model}@{dataset}", 'roc.pdf'), format="pdf", bbox_inches="tight")
plt.tight_layout()
# plt.legend(loc='lower right')
plt.show()

