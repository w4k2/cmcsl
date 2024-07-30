"""
Histograms of mean features depending on normalization + runs od CMCSL.
"""

import numpy as np
from tabulate import tabulate
from utils import t_test_corrected, cv52cft
from scipy.stats import rankdata, ranksums, wilcoxon


alfa = .05

root = "../../mm-datasets/data_extracted/"
topics = ["HR", "DS", "HM", "BW", "AM", "FS", "FW", "MS", "FM", "AT", 
          "HMSWW", "FMSW", "FMNSSW", "FFW", "CDFS", "AFHMS", "ACMD", "ACCDDHMRS", "ABHMW", "ABFHSS"]

modalities = [
    ["img", "txt"],
]
modalities_names = [
    ["Visual", "Text"],
]

titles = ["RAW", "L2 Normalized", "Standarized", "MinMax", "L2 Normalized + Standarized"]
titles = ["RAW", "L2", "STD", "MM", "L2STD"]

alg_names = [
    "full", 
    "seed", 
    "single",
    "cross",
    ]

preproc_names = [
    "off",
    "norm",
    "stand",
    "minmax",
    "both1",
    ]

n_clfs = len(preproc_names)

# PREPROC x TOPIC x ALG x TIMES x FOLDS
scores_m1 = np.load("scores/experiment_2_m1_52.npy")
scores_m2 = np.load("scores/experiment_2_m2_52.npy")
# MODALITY x PREPROC x TOPIC x ALG x TIMES x FOLDS
scores_all = np.stack((scores_m1, scores_m2))

# PREPROC x TOPIC x ALG x TIMES x FOLDS
scores_m1_more = np.load("scores/experiment_2_m1_52_more.npy")
scores_m2_more = np.load("scores/experiment_2_m2_52_more.npy")
# MODALITY x PREPROC x TOPIC x ALG x TIMES x FOLDS
scores_all_more = np.stack((scores_m1_more, scores_m2_more))


# MODALITY x PREPROC x TOPIC x ALG x TIMES x FOLDS
scores_all = np.concatenate((scores_all, scores_all_more), axis=1)
# MODALITY x PREPROC x TOPIC x ALG x TIMES x FOLDS
scores_all = scores_all[:, [0, 1, 2, 4, 3]]
# MODALITY x PREPROC x TOPIC x TIMES x FOLDS
scores_all = scores_all[:, :, :, 3]
# MODALITY x PREPROC x TOPIC x FOLDS
scores_all = np.mean(scores_all, axis=3)
print(scores_all.shape)
scores_all_binary = scores_all[:, :, :10]
scores_all_multi = scores_all[:, :, 10:]
# MODALITY x PREPROC x TOPIC
# mean_scores_all = np.mean(scores_all, axis=3)
# print(mean_scores_all.shape)
# mean_scores_all_binary = mean_scores_all[:, :, :10]
# mean_scores_all_multi = mean_scores_all[:, :, 10:]

# Binary
all = []
ranks_ = []
wils = []
for data_id, data in enumerate(topics[:10]):
    # MODALITY x PREPROC x FOLDS
    data_scores = scores_all_binary[:, :, data_id]
    for modality_id, modality in enumerate(modalities_names[0]):
        # PREPROC x FOLDS
        modality_scores = data_scores[modality_id]
        ranks_.append(rankdata(np.mean(modality_scores, axis=1)))
        wils.append(np.mean(modality_scores, axis=1))
        
        all.append(["%s"% data] + ["%s"% modality] + ["%.3f" % score for score in np.mean(modality_scores, axis=1)])
        
        # t-test corrected
        T, p = np.array([[cv52cft(modality_scores[i],
                               modality_scores[j]) if i != j else (0.0, 1.0)
                              for i in range(n_clfs)]
                             for j in range(n_clfs)]
                        ).swapaxes(0, 2)
        mean_adv = np.mean(modality_scores, axis=1) < np.mean(modality_scores, axis=1)[:, np.newaxis]
        stat_adv = p < alfa
        
        _ = np.where(stat_adv * mean_adv)
        conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]
        
        all.append([''] + [''] + [", ".join(["%i" % i for i in c])
                                         if len(c) > 0 and len(c) < n_clfs-1 else ("all" if len(c) == n_clfs-1 else "---")
                                         for c in conclusions])
        
# Wilcoxon
# DATASETS x MODALITIES x CLF
ranks = np.array(wils)
# print(ranks, ranks.shape)
mean_ranks = np.mean(ranks_, axis=0)

w_statistic = np.zeros((n_clfs, n_clfs))
p = np.zeros((n_clfs, n_clfs))

for i in range(n_clfs):
    for j in range(n_clfs):
        w_statistic[i, j], p[i, j] = wilcoxon(ranks.T[i], ranks.T[j], zero_method="zsplit", alternative="greater")

_ = np.where((p < alfa) * (w_statistic > 0))
conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]

all.append(['Average rank'] + [''] + ["%.3f" % v for v in mean_ranks])
all.append([''] + [''] + [", ".join(["%i" % i for i in c])
                             if len(c) > 0 and len(c) < n_clfs-1 else ("all" if len(c) == n_clfs-1 else "---")
                             for c in conclusions])
        
print(tabulate(all, headers=["Dataset"] + ["M"] + titles, floatfmt=".3f", tablefmt="latex_booktabs"))

# Multiclass
all = []
ranks_ = []
wils = []
for data_id, data in enumerate(topics[10:]):
    # MODALITY x PREPROC x FOLDS
    data_scores = scores_all_multi[:, :, data_id]
    for modality_id, modality in enumerate(modalities_names[0]):
        # PREPROC x FOLDS
        modality_scores = data_scores[modality_id]
        ranks_.append(rankdata(np.mean(modality_scores, axis=1)))
        wils.append(np.mean(modality_scores, axis=1))
        
        all.append(["%s"% data] + ["%s"% modality] + ["%.3f" % score for score in np.mean(modality_scores, axis=1)])
        
        # t-test corrected
        T, p = np.array([[cv52cft(modality_scores[i],
                               modality_scores[j]) if i != j else (0.0, 1.0)
                              for i in range(n_clfs)]
                             for j in range(n_clfs)]
                        ).swapaxes(0, 2)
        mean_adv = np.mean(modality_scores, axis=1) < np.mean(modality_scores, axis=1)[:, np.newaxis]
        stat_adv = p < alfa
        
        _ = np.where(stat_adv * mean_adv)
        conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]
        
        all.append([''] + [''] + [", ".join(["%i" % i for i in c])
                                         if len(c) > 0 and len(c) < n_clfs-1 else ("all" if len(c) == n_clfs-1 else "---")
                                         for c in conclusions])
        
# Wilcoxon
# DATASETS x MODALITIES x CLF
ranks = np.array(wils)
# print(ranks, ranks.shape)
mean_ranks = np.mean(ranks_, axis=0)

w_statistic = np.zeros((n_clfs, n_clfs))
p = np.zeros((n_clfs, n_clfs))

for i in range(n_clfs):
    for j in range(n_clfs):
        w_statistic[i, j], p[i, j] = wilcoxon(ranks.T[i], ranks.T[j], zero_method="zsplit", alternative="greater")

_ = np.where((p < alfa) * (w_statistic > 0))
conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]

all.append(['Average rank'] + [''] + ["%.3f" % v for v in mean_ranks])
all.append([''] + [''] + [", ".join(["%i" % i for i in c])
                             if len(c) > 0 and len(c) < n_clfs-1 else ("all" if len(c) == n_clfs-1 else "---")
                             for c in conclusions])
        
print(tabulate(all, headers=["Dataset"] + ["M"] + titles, floatfmt=".3f", tablefmt="latex_booktabs"))