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
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold


root = "../../mm-datasets/data_extracted/"
datasets = [
    # "kinetics400",
    "mmIMDb",
    ]
topics = [
    # ["sport", "instruments", "riding", "dancing", "eating"],
    ["HR", "DS", "HM", "BW", "AM", "FS", "FW", "MS", "FM", "AT", 
     "HMSWW", "FMSW", "FMNSSW", "FFW", "CDFS", "AFHMS", "ACMD", "ACCDDHMRS", "ABHMW", "ABFHSS"],
]
modalities = [
    # ["video", "audio", "y"],
    ["img", "txt", "y"],
]

 # Base CLF
# base_clf = GaussianNB()

base_clfs = [GaussianNB(), LogisticRegression(random_state=1410, max_iter=5000), DecisionTreeClassifier(random_state=1410)]

clf_names = ["gnb", "lg", "cart"]

n_times = np.array([i+1 for i in range(20)])

algorithms = [mmBaseline]
alg_names = [
    "full late fusion"
    ]

# DATASETS x ALG X TIMES x FOLDS x BASE
scores_lf = np.zeros((len(topics[0]), len(algorithms), n_times.shape[0], 10, 3))

# For each modality get clusters and distances
for dataset_id, dataset in tqdm(enumerate(datasets), disable=True):
    for topic_id, topic in tqdm(enumerate(topics[dataset_id]), disable=False, total=len(topics[dataset_id])):
        X_m1_path = root + "%s/%s_%s_%s_weighted_30.npy" % (dataset, dataset, topic, modalities[dataset_id][0]) if dataset_id == 0 else root + "%s/weighted_%s_%s_%s.npy" % (dataset, dataset, modalities[dataset_id][0], topic)
        X_m2_path = root + "%s/%s_%s_%s.npy" % (dataset, dataset, topic, modalities[dataset_id][1]) if dataset_id == 0 else root + "%s/weighted_%s_%s_%s.npy" % (dataset, dataset, modalities[dataset_id][1], topic)
        y_path = root + "%s/%s_%s_%s.npy" % (dataset, dataset, topic, modalities[dataset_id][2]) if dataset_id == 0 else root + "%s/weighted_%s_%s_%s.npy" % (dataset, dataset, modalities[dataset_id][2], topic)
        
        X_m1 = np.load(X_m1_path)
        X_m2 = np.load(X_m2_path)
        y = np.load(y_path)
        
        X_concatenated = np.concatenate((X_m1, X_m2), axis=1)
        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1410)
        
        for fold_id, (train, test) in tqdm(enumerate(rskf.split(X_concatenated, y)), total=10):
            X_m1_train, X_m2_train = X_concatenated[train][:, :X_m1.shape[1]], X_concatenated[train][:, X_m1.shape[1]:]
            X_m1_test, X_m2_test = X_concatenated[test][:, :X_m1.shape[1]], X_concatenated[test][:, X_m1.shape[1]:]
            y_train, y_test = y[train], y[test]
            
            for times_id, times in enumerate(n_times):
                for algorithm_id in range(len(algorithms)):
                    for clf_id, base_clf in enumerate(base_clfs):
                        clf = algorithms[algorithm_id](clone(base_clf)) if algorithm_id == 0 else  algorithms[algorithm_id](clone(base_clf), random_state=1410, times=times, mode=alg_names[algorithm_id])
                        
                        clf.fit([X_m1_train, X_m2_train], y_train)
                        
                        pred_lf = clf.predict_combined([X_m1_test, X_m2_test])
                        
                        scores_lf[topic_id, algorithm_id, times_id, fold_id, clf_id] = balanced_accuracy_score(y_test, pred_lf)
                        
        np.save("scores/experiment_3_52_late_fusion_all_clfs", scores_lf)