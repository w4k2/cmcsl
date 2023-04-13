"""
Histograms of mean features depending on normalization.
"""

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import Normalizer, StandardScaler
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA


root = "../../mm-datasets/data_extracted/"
datasets = ["mmIMDb"]
topics = [
    ["HR", "DS", "HM", "BW", "AM", "FS", "FW", "MS", "FM", "AT", 
     "HMSWW", "FMSW", "FMNSSW", "FFW", "CDFS", "AFHMS", "ACMD", "ACCDDHMRS", "ABHMW", "ABFHSS"],
]
modalities = [
    ["img", "txt", "y"],
]

titles = ["RAW", "Normalized", "Standarized", "Normalized + Standarized"]

# For each modality get clusters and distances
for dataset_id, dataset in tqdm(enumerate(datasets), disable=True):
    for topic_id, topic in tqdm(enumerate(topics[dataset_id]), disable=False, total=len(topics[dataset_id])):
        X_m1_path = root + "%s/%s_%s_%s_weighted_30.npy" % (dataset, dataset, topic, modalities[dataset_id][0])
        X_m2_path = root + "%s/%s_%s_%s.npy" % (dataset, dataset, topic, modalities[dataset_id][1])
        y_path = root + "%s/%s_%s_%s.npy" % (dataset, dataset, topic, modalities[dataset_id][2])
        
        X_m1 = np.load(X_m1_path)
        X_m2 = np.load(X_m2_path)
        y = np.load(y_path)
        
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        
        ros = RandomOverSampler(random_state=1410)
        X_m1, y_m1 = ros.fit_resample(X_m1, y)
        X_m2, y_m2 = ros.fit_resample(X_m2, y)
        
        
        # m1
        max_counts = []
        for i in range(4):
            classes = np.unique(y)
            
            if i == 0:
                mean_X_m1 = X_m1
            elif i == 1:
                norm = Normalizer()
                mean_X_m1 = norm.fit_transform(X_m1)
            elif i == 2:
                stand = StandardScaler()
                mean_X_m1 = stand.fit_transform(X_m1)
            elif i == 3:
                norm = Normalizer()
                mean_X_m1 = norm.fit_transform(X_m1)
                stand = StandardScaler()
                mean_X_m1 = stand.fit_transform(mean_X_m1)
            # mean_X_m1 = X_m1
            mean_X_m1 = np.mean(mean_X_m1, axis=1)
            for class_id in classes:
                counts, bins = np.histogram(mean_X_m1[y_m1 == class_id], bins=64)
                ax[0, i].stairs(counts, bins)
                ax[0, i].set_title("IMG %s" % (titles[i]), fontsize = 16)
                ax[0, i].spines[['right', 'top']].set_visible(False)
                ax[0, i].grid((.7, .7, .7), ls=":")
                ax[0, i].set_ylabel("Counts", fontsize = 12)
                ax[0, i].set_xlabel("Bins", fontsize = 12)
                
                max_counts.append(np.max(counts))
            # ax[0, i].set_ylim(0.0, np.max(max_counts))
        # m2
        for i in range(4):
            classes = np.unique(y)
            
            if i == 0:
                mean_X_m2 = X_m2
            elif i == 1:
                norm = Normalizer()
                mean_X_m2 = norm.fit_transform(X_m2)
            elif i == 2:
                stand = StandardScaler()
                mean_X_m2 = stand.fit_transform(X_m2)
            elif i == 3:
                norm = Normalizer()
                mean_X_m2 = norm.fit_transform(X_m2)
                stand = StandardScaler()
                mean_X_m2 = stand.fit_transform(mean_X_m2)
                
            mean_X_m2 = np.mean(mean_X_m2, axis=1)
            # max_counts = []
            for class_id in classes:
                counts, bins = np.histogram(mean_X_m2[y_m2 == class_id], bins=32)
                ax[1, i].stairs(counts, bins)
                ax[1, i].set_title("TXT %s" % (titles[i]), fontsize = 16)
                ax[1, i].spines[['right', 'top']].set_visible(False)
                ax[1, i].grid((.7, .7, .7), ls=":")
                ax[1, i].set_ylabel("Counts", fontsize = 12)
                ax[1, i].set_xlabel("Bins", fontsize = 12)
                
                max_counts.append(np.max(counts))
            # ax[1, i].set_ylim(0.0, np.max(max_counts))
            
        ax = ax.ravel()
        for i in range(ax.shape[0]):
            round_to = round(50 * round(np.max(max_counts) / 50) + 50, -1)
            ax[i].set_ylim(0.0, round_to)
        # print(max_counts)
        # exit()
        plt.tight_layout()
        plt.savefig("figures/ex2/hist_%s.png" % topic)