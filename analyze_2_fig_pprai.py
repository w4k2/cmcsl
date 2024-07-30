"""
Histograms of mean features depending on normalization + runs od CMCSL.
"""

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
import matplotlib


matplotlib.rcParams.update({'font.size': 13, "font.family" : "monospace"})


root = "../../mm-datasets/data_extracted/"
datasets = ["mmIMDb"]
topics = [
    ["HR", "DS", "HM", "BW", "AM", "FS", "FW", "MS", "FM", "AT", 
     "HMSWW", "FMSW", "FMNSSW", "FFW", "CDFS", "AFHMS", "ACMD", "ACCDDHMRS", "ABHMW", "ABFHSS"],
]
modalities = [
    ["img", "txt", "y"],
]
modalities_names = [
    ["VISUAL", "TEXT", "y"],
]

titles = ["RAW", "L2", "STD", "L2STD", "L2STD"]

titles = ["L2","L2STD"]

alg_names = [
    "FULL", 
    "PRE", 
    "UNI",
    "CMCSL",
    ]

# preproc_names = [
#     "off",
#     "norm",
#     "stand",
#     "both"
#     ]

# preproc_names = [
#     "minmax",
#     "both2",
#     "both3",
#     "both4",
#     "three",
#     ]

preproc_names = [
    # "off",
    "norm",
    # "stand",
    # "minmax",
    "both1",
    # "both2",
    # "both3",
    # "both4",
    # "three",
    ]
# titles = preproc_names

ls = ["-", ":", "--", (10, (20, 40))]
lw = 1.5

n_times = np.array([i+1 for i in range(20)])

# PREPROC x TOPIC x ALG x TIMES x FOLDS
scores_m1 = np.load("scores/experiment_2_m1_52.npy")
scores_m2 = np.load("scores/experiment_2_m2_52.npy")
# PREPROC x TOPIC x ALG x TIMES x FOLDS
# scores_m1_more = np.load("scores/experiment_2_m1_52_more.npy")
# scores_m2_more = np.load("scores/experiment_2_m2_52_more.npy")
# # PREPROC x TOPIC x ALG x TIMES x FOLDS
# scores_m1 = np.concatenate((scores_m1, scores_m1_more), axis=0)
# scores_m2 = np.concatenate((scores_m2, scores_m2_more), axis=0)

scores_m1 = scores_m1[[1, 3]]
scores_m2 = scores_m2[[1, 3]]

print(scores_m1.shape)
print(scores_m2.shape)

# PREPROC x TOPIC x ALG x TIMES
scores_m1 = np.mean(scores_m1, axis=4)
scores_m2 = np.mean(scores_m2, axis=4)

# For each modality get clusters and distances
for dataset_id, dataset in tqdm(enumerate(datasets), disable=True):
    for topic_id, topic in tqdm(enumerate(topics[dataset_id]), disable=False, total=len(topics[dataset_id])):
        # PREPROC x ALG x TIMES
        topic_scores_m1 = scores_m1[:, topic_id]
        topic_scores_m2 = scores_m2[:, topic_id]
        
        X_m1_path = root + "%s/%s_%s_%s_weighted_30.npy" % (dataset, dataset, topic, modalities[dataset_id][0])
        X_m2_path = root + "%s/%s_%s_%s.npy" % (dataset, dataset, topic, modalities[dataset_id][1])
        y_path = root + "%s/%s_%s_%s.npy" % (dataset, dataset, topic, modalities[dataset_id][2])
        
        X_m1 = np.load(X_m1_path)
        X_m2 = np.load(X_m2_path)
        y = np.load(y_path)
        
        fig, ax = plt.subplots(2, len(preproc_names), figsize=(10, 10))
        
        ros = RandomOverSampler(random_state=1410)
        X_m1, y_m1 = ros.fit_resample(X_m1, y)
        X_m2, y_m2 = ros.fit_resample(X_m2, y)
        
        
        # m1 hist
        max_counts = []
        for i in range(len(preproc_names)):
            classes = np.unique(y)
            
            # if i == 0:
            #     mean_X_m1 = X_m1
            if i == 0:
                norm = Normalizer()
                mean_X_m1 = norm.fit_transform(X_m1)
            # elif i == 2:
            #     stand = StandardScaler()
            #     mean_X_m1 = stand.fit_transform(X_m1)
            # elif i == 3:
            #     minmax = MinMaxScaler()
            #     mean_X_m1 = minmax.fit_transform(X_m1)
            elif i == 1:
                norm = Normalizer()
                mean_X_m1 = norm.fit_transform(X_m1)
                stand = StandardScaler()
                mean_X_m1 = stand.fit_transform(mean_X_m1)
            # mean_X_m1 = X_m1
            mean_X_m1 = np.mean(mean_X_m1, axis=1)
            for class_id in classes:
                counts, bins = np.histogram(mean_X_m1[y_m1 == class_id], bins=32)
                ax[0, i].stairs(counts, bins)
                ax[0, i].set_title("VISUAL %s" % (titles[i]), fontsize = 15)
                ax[0, i].spines[['right', 'top']].set_visible(False)
                ax[0, i].grid((.7, .7, .7), ls=":")
                ax[0, i].set_ylabel("Counts", fontsize = 12)
                ax[0, i].set_xlabel("Bins", fontsize = 12)
                
                max_counts.append(np.max(counts))
            # ax[0, i].set_ylim(0.0, np.max(max_counts))
        
        # m1 run
        for preproc_id, preproc in enumerate(preproc_names):
            preproc_scores_m1 = topic_scores_m1[preproc_id]
            preproc_scores_m2 = topic_scores_m2[preproc_id]
            for algorithm_id in range(len(alg_names)):
                alg_scores_m1 = preproc_scores_m1[algorithm_id]
                alg_scores_m2 = preproc_scores_m2[algorithm_id]
                
                if algorithm_id == len(alg_names)-1:
                    ax[1, preproc_id].plot(n_times, alg_scores_m1, c="red", ls="-.", lw=lw, label = "%s %s" % (modalities_names[0][0], alg_names[algorithm_id]))
                    ax[1, preproc_id].plot(n_times, alg_scores_m2, c="blue", ls="-.", lw=lw, label = "%s %s" % (modalities_names[0][1], alg_names[algorithm_id]))
                else:
                    ax[1, preproc_id].plot(n_times, alg_scores_m1, c="red", lw=.8, ls=ls[algorithm_id], label = "%s %s" % (modalities_names[0][0], alg_names[algorithm_id]))
                    ax[1, preproc_id].plot(n_times, alg_scores_m2, c="blue", lw=.8, ls=ls[algorithm_id], label = "%s %s" % (modalities_names[0][1], alg_names[algorithm_id]))
                    
                ax[1, preproc_id].spines[['right', 'top']].set_visible(False)
                ax[1, preproc_id].grid((.7, .7, .7), ls=":")
                ax[1, preproc_id].set_ylabel("Balanced accuracy", fontsize = 12)
                ax[1, preproc_id].set_xlabel("#samples for each class", fontsize = 12)
                ax[1, preproc_id].set_xlim(n_times[0], n_times[-1])
                ax[1, preproc_id].set_xticks([1, 5, 10, 15, 20])
            
            
        # # m2
        # for i in range(len(preproc_names)):
        #     classes = np.unique(y)
            
        #     if i == 0:
        #         mean_X_m2 = X_m2
        #     elif i == 1:
        #         norm = Normalizer()
        #         mean_X_m2 = norm.fit_transform(X_m2)
        #     elif i == 2:
        #         stand = StandardScaler()
        #         mean_X_m2 = stand.fit_transform(X_m2)
        #     elif i == 3:
        #         minmax = MinMaxScaler()
        #         mean_X_m2 = minmax.fit_transform(X_m2)
        #     elif i == 4:
        #         norm = Normalizer()
        #         mean_X_m2 = norm.fit_transform(X_m2)
        #         stand = StandardScaler()
        #         mean_X_m2 = stand.fit_transform(mean_X_m2)
                
        #     mean_X_m2 = np.mean(mean_X_m2, axis=1)
        #     # max_counts = []
        #     for class_id in classes:
        #         counts, bins = np.histogram(mean_X_m2[y_m2 == class_id], bins=32)
        #         ax[1, i].stairs(counts, bins)
        #         ax[1, i].set_title("TEXT %s" % (titles[i]), fontsize = 15)
        #         ax[1, i].spines[['right', 'top']].set_visible(False)
        #         ax[1, i].grid((.7, .7, .7), ls=":")
        #         ax[1, i].set_ylabel("Counts", fontsize = 12)
        #         ax[1, i].set_xlabel("Bins", fontsize = 12)
                
        #         max_counts.append(np.max(counts))
            
        # ax = ax.ravel()
        for i in range(len(preproc_names)):
            round_to = round(50 * round(np.max(max_counts) / 50) + 50, -1)
            ax[0, i].set_ylim(0.0, round_to)
            # ax[1, i].set_ylim(0.0, round_to)
            ax[1, i].set_ylim(0.5, 1.0)
        
        
        plt.tight_layout()
        fig.subplots_adjust(bottom=.15)
        handles, labels = ax[1, 1].get_legend_handles_labels()
        lgnd = fig.legend(handles, labels, ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=16, loc='lower center')
        
        
        plt.savefig("figures/ex2/whole_%s_more_pprai24.png" % topic)
        plt.savefig("figures/ex2/whole_%s_more_pprai24.eps" % topic)
        # exit()