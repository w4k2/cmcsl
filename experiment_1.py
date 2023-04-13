"""
Base classifiers trained on deep features.
"""

from utils import  mmBaseline, CMSL
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import Normalizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier


root = "../../mm-datasets/data_extracted/"
datasets = ["mmIMDb"]
topics = [
    ["HR", "DS", "HM", "BW", "AM", "FS", "FW", "MS", "FM", "AT", 
     "HMSWW", "FMSW", "FMNSSW", "FFW", "CDFS", "AFHMS", "ACMD", "ACCDDHMRS", "ABHMW", "ABFHSS"],
]
modalities = [
    ["img", "txt", "y"],
]

base_clfs = [
    GaussianNB(),
    LogisticRegression(random_state=1410, max_iter=5000),
    DecisionTreeClassifier(random_state=1410)
]

# DATASETS x MODALITIES x FOLDS x CLF
scores = np.zeros((20, 2, 10, 3))

# For each modality get clusters and distances
for dataset_id, dataset in tqdm(enumerate(datasets), disable=True):
    for topic_id, topic in tqdm(enumerate(topics[dataset_id]), disable=False, total=len(topics[dataset_id])):
        X_m1_path = root + "%s/%s_%s_%s_weighted_30.npy" % (dataset, dataset, topic, modalities[dataset_id][0])
        X_m2_path = root + "%s/%s_%s_%s.npy" % (dataset, dataset, topic, modalities[dataset_id][1])
        y_path = root + "%s/%s_%s_%s.npy" % (dataset, dataset, topic, modalities[dataset_id][2])
        
        X_m1 = np.load(X_m1_path)
        X_m2 = np.load(X_m2_path)
        y = np.load(y_path)
        
        X_concatenated = np.concatenate((X_m1, X_m2), axis=1)
        
        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1410)
        
        for fold_id, (train, test) in tqdm(enumerate(rskf.split(X_concatenated, y)), total=10):
            X_m1_train, X_m2_train = X_concatenated[train][:, :X_m1.shape[1]], X_concatenated[train][:, X_m1.shape[1]:]
            X_m1_test, X_m2_test = X_concatenated[test][:, :X_m1.shape[1]], X_concatenated[test][:, X_m1.shape[1]:]
            y_train, y_test = y[train], y[test]
            
            _, counts = np.unique(y_train, return_counts=True)
            class_weights = np.array([1-(weight/np.sum(counts)) for weight in counts])
            sample_weights = class_weights[y_train]
            
            for clf_id, clf in enumerate(base_clfs):
                clf_m1 = clone(clf).fit(X_m1_train, y_train, sample_weight = sample_weights)
                clf_m2 = clone(clf).fit(X_m2_train, y_train, sample_weight = sample_weights)
                
                y_pred_m1 = clf_m1.predict(X_m1_test)
                y_pred_m2 = clf_m2.predict(X_m2_test)
                scores[topic_id, 0, fold_id, clf_id] = balanced_accuracy_score(y_test, y_pred_m1)
                scores[topic_id, 1, fold_id, clf_id] = balanced_accuracy_score(y_test, y_pred_m2)
                
np.save("scores/experiment_1_52", scores)