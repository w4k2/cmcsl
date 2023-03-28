"""

"""

from utils import  mmBaseline, CMSL
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import Normalizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier


root = "../../mm-datasets/data_extracted/"
datasets = [
    "kinetics400",
    "mmIMDb",
    ]
topics = [
    ["sport", "instruments", "riding", "dancing", "eating"],
    ["HR", "ABHMW", "CDFS", "FMNSSW", "ACCDDHMRS"],
]
modalities = [
    ["video", "audio", "y"],
    ["img", "txt", "y"],
]

# For each modality get clusters and distances
for dataset_id, dataset in tqdm(enumerate(datasets), disable=True):
    for topic_id, topic in tqdm(enumerate(topics[dataset_id]), disable=False, total=len(topics[dataset_id])):
        X_m1_path = root + "%s/%s_%s_%s_weighted.npy" % (dataset, dataset, topic, modalities[dataset_id][0]) if dataset_id == 1 else root + "%s/weighted_%s_%s_%s.npy" % (dataset, dataset, modalities[dataset_id][0], topic)
        X_m2_path = root + "%s/%s_%s_%s.npy" % (dataset, dataset, topic, modalities[dataset_id][1]) if dataset_id == 1 else root + "%s/weighted_%s_%s_%s.npy" % (dataset, dataset, modalities[dataset_id][1], topic)
        y_path = root + "%s/%s_%s_%s.npy" % (dataset, dataset, topic, modalities[dataset_id][2]) if dataset_id == 1 else root + "%s/weighted_%s_%s_%s.npy" % (dataset, dataset, modalities[dataset_id][2], topic)
        
        X_m1 = np.load(X_m1_path)
        X_m2 = np.load(X_m2_path)
        y = np.load(y_path)
        
        X_m1_train, X_m1_test, X_m2_train, X_m2_test, y_train, y_test = train_test_split(X_m1, X_m2, y, test_size=0.8, random_state=1410, stratify=y)
        
        # Base CLF
        base_clf = GaussianNB()
        
        n_times = np.array([i+1 for i in range(20)])
        
        algorithms = [mmBaseline, 
                      CMSL,
                      CMSL,
                      CMSL,
                      ]
        alg_names = [
            "full", 
            "seed", 
            "single",
            "cross",
            ]
        ls = [":", "--", "-.", "-"]
        scores_m1 = np.zeros((len(algorithms), n_times.shape[0]))
        scores_m2 = np.zeros((len(algorithms), n_times.shape[0]))
        
        for times_id, times in enumerate(n_times):
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks([i for i in n_times])
            ax.set_xlim(n_times[0], n_times[-1])
            ax.spines[['right', 'top']].set_visible(False)
            
            for algorithm_id in range(len(algorithms)):
                clf = algorithms[algorithm_id](clone(base_clf), normalized=True) if algorithm_id == 0 else  algorithms[algorithm_id](clone(base_clf), random_state=1410, times=times, mode=alg_names[algorithm_id])
                
                clf.fit([X_m1_train, X_m2_train], y_train)
                
                pred_m1, pred_m2 = clf.predict([X_m1_test, X_m2_test])
                scores_m1[algorithm_id, times_id], scores_m2[algorithm_id, times_id] = balanced_accuracy_score(y_test, pred_m1), balanced_accuracy_score(y_test, pred_m2)
                
                ax.plot(n_times[:times_id+1], scores_m1[algorithm_id,:times_id+1], c="red", ls=ls[algorithm_id], label = "%s %s %.3f" % (modalities[dataset_id][0], alg_names[algorithm_id], np.mean(scores_m1[algorithm_id,:times_id+1])))
                ax.plot(n_times[:times_id+1], scores_m2[algorithm_id,:times_id+1], c="blue", ls=ls[algorithm_id], label = "%s %s %.3f" % (modalities[dataset_id][1], alg_names[algorithm_id], np.mean(scores_m2[algorithm_id,:times_id+1])))
            
            plt.title("%s %s %i" % (dataset, topic, np.unique(y).shape[0]))
            plt.tight_layout()
            plt.legend(fontsize = 14, frameon=False)
            plt.savefig("foo.png", dpi=200)
            # plt.savefig("figures/sl_%s_%s.png" % (dataset, topic), dpi=200)
            plt.close()
        # exit()