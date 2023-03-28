import numpy as np
from tabulate import tabulate


# DATASETS x MODALITIES x FOLDS x CLF
scores = np.load("scores/experiment_1.npy")
print(scores.shape)

topics = [
    ["HR", "ABHMW", "CDFS", "FMNSSW"],
]

clfs = [
    "GNB", 
    "LogR", 
    "CART"]

modalities = [
    ["img", "txt"],
]

all = []
for data_id, data in enumerate(topics[0]):
    print(data)
    data_scores = scores[data_id]
    for modality_id, modality in enumerate(modalities[0]):
        print(modality)
        modality_scores = data_scores[modality_id]
        all.append(["%s"% data] + ["%s"% modality] + [score for score in np.mean(modality_scores, axis=0)])
print(tabulate(all, headers=["Dataset"] + ["Modality"] + clfs, floatfmt=".3f"))
            
            