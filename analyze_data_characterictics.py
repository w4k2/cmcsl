"""
Table with data characteristics
"""

import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from sklearn.model_selection import train_test_split


root = "../../mm-datasets/data_extracted/"
datasets = [
    "mmIMDb",
    ]
topics = [
    ["HR", "DS", "HM", "BW", "AM", "FS", "FW", "MS", "FM", "AT", 
     "HMSWW", "FMSW", "FMNSSW", "FFW", "CDFS", "AFHMS", "ACMD", "ACCDDHMRS", "ABHMW", "ABFHSS"],
]
modalities = [
    ["img", "txt", "y"],
]

class_names = [ 
    # # Binary
    # # HR
    ["Horror", "Romance"],
    # # DS
    ["Documentary", "Sci-Fi"],
    # # HM
    ["History", "Music"],
    # # BW
    ["Biography", "Western"],
    # # AM
    ["Animation", "Musical"],
    # # FS
    ["Film-Noir", "Short"],
    # # FW
    ["Family", "War"],
    # # MS
    ["Musical", "Sport"],
    # # FM
    ["Fantasy", "Mystery"],
    # # AT
    ["Adventure", "Thriller"],
    # # Multiclass
    # HMSWW
    ["History", "Musical", "Sci-Fi", "War", "Western"],
    # FMSW
    ["Family", "Musical", "Sci-Fi", "War"],
    # FMNSSW
    ["Film-Noir", "Musical", "News", "Short", "Sport", "Western"],
    # FFW
    ["Fantasy", "Film-Noir", "Western"],
    # CDFS
    ["Crime", "Documentary", "Fantasy", "Sci-Fi"],
    # AFHMS
    ["Animation", "Fantasy", "History", "Mystery", "Sport"],
    # ACMD
    ["Adventure", "Crime", "Music", "Documentary"],
    # ACCDDHMRS
    ['Action', 'Comedy', 'Crime','Documentary', 'Drama', 'Horror', 'Mystery', 'Romance', 'Sci-Fi'],
    # ABHMW
    ["Animation", "Biography", "History", "Music", "War"],
    # ABFHSS
    ["Action", "Biography", "Family", "Horror", "Short", "Sport"]]

# For each modality get clusters and distances
table = []
headers = ["Abbreviation", "Classes & counts", "#samples"]

for dataset_id, dataset in tqdm(enumerate(datasets), disable=True):
    for topic_id, topic in tqdm(enumerate(topics[dataset_id]), disable=False, total=len(topics[dataset_id])):
        X_m1_path = root + "%s/%s_%s_%s_weighted_30.npy" % (dataset, dataset, topic, modalities[dataset_id][0]) if dataset_id == 0 else root + "%s/weighted_%s_%s_%s.npy" % (dataset, dataset, modalities[dataset_id][0], topic)
        X_m2_path = root + "%s/%s_%s_%s.npy" % (dataset, dataset, topic, modalities[dataset_id][1]) if dataset_id == 0 else root + "%s/weighted_%s_%s_%s.npy" % (dataset, dataset, modalities[dataset_id][1], topic)
        y_path = root + "%s/%s_%s_%s.npy" % (dataset, dataset, topic, modalities[dataset_id][2]) if dataset_id == 0 else root + "%s/weighted_%s_%s_%s.npy" % (dataset, dataset, modalities[dataset_id][2], topic)
        
        X_m1 = np.load(X_m1_path)
        X_m2 = np.load(X_m2_path)
        y = np.load(y_path)
        
        X_concatenated = np.concatenate((X_m1, X_m2), axis=1)
        
        X_train, X_extract, y_train, y_extract = train_test_split(
            X_concatenated, y, test_size=.8, random_state=1410, stratify=y, shuffle=True)
    
        _, counts = np.unique(y_extract, return_counts=True)
        
        class_names_counts = []
        for id, i in enumerate(class_names[topic_id]):
            class_names_counts.append("%s (%i)" % (i, counts[id]))
        
        table.append(["%s" % topic] + [", ".join(class_names_counts)] + [y_extract.shape[0]])
        

print(tabulate(table, headers=headers, tablefmt="latex_booktabs"))