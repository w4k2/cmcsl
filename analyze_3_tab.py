"""

"""
import numpy as np
from tabulate import tabulate
from utils import t_test_corrected
from scipy.stats import rankdata, ranksums

alfa = .05

root = "../../mm-datasets/data_extracted/"
datasets = [
    "mmIMDb",
    ]
topics = ["HR", "DS", "HM", "BW", "AM", "FS", "FW", "MS", "FM", "AT", 
          "HMSWW", "FMSW", "FMNSSW", "FFW", "CDFS", "AFHMS", "ACMD", "ACCDDHMRS", "ABHMW", "ABFHSS"]

modalities = [
    ["img", "txt"],
]

alg_names = [
    "full", 
    "seed", 
    "single",
    "cross",
    ]

n_clfs = len(alg_names)

# TOPIC x ALG x TIMES x FOLDS
scores_m1 = np.load("scores/experiment_3_m1_52.npy")
scores_m2 = np.load("scores/experiment_3_m2_52.npy")
scores_m1_mc = np.load("scores/experiment_3_m1_52_mc.npy")
scores_m2_mc = np.load("scores/experiment_3_m2_52_mc.npy")
# MODALITY x TOPIC x ALG x TIMES x FOLDS
scores_all = np.stack((scores_m1, scores_m2))
scores_all_mc = np.stack((scores_m1_mc, scores_m2_mc))
scores_all = np.concatenate((scores_all.reshape(2, 20, 4, 20, 10, 1), scores_all_mc), axis=5)

# MODALITY x TOPIC x ALG x FOLDS
scores_all = np.mean(scores_all, axis=3)
scores_all = scores_all[:, :, :, :, 0]
scores_all_binary = scores_all[:, :10]
scores_all_multi = scores_all[:, 10:]
print(scores_all_binary, scores_all_binary.shape)

# Binary
all = []
ranks = []
for data_id, data in enumerate(topics[:10]):
    # MODALITY x ALG x FOLDS
    data_scores = scores_all_binary[:, data_id]
    for modality_id, modality in enumerate(modalities[0]):
        # ALG x FOLDS
        modality_scores = data_scores[modality_id]
        ranks.append(rankdata(np.mean(modality_scores[1:], axis=1)))
        
        all.append(["%s"% data] + ["%s"% modality] + ["%.3f" % score for score in np.mean(modality_scores, axis=1)])
        
        # t-test corrected
        T, p = np.array([[t_test_corrected(modality_scores[[1, 2, 3]][i],
                               modality_scores[[1, 2, 3]][j]) if i != j else (0.0, 1.0)
                              for i in range(n_clfs-1)]
                             for j in range(n_clfs-1)]
                        ).swapaxes(0, 2)
        mean_adv = np.mean(modality_scores[[1, 2, 3]], axis=1) < np.mean(modality_scores[[1, 2, 3]], axis=1)[:, np.newaxis]
        stat_adv = p < alfa
        
        _ = np.where(stat_adv * mean_adv)
        conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]
        
        all.append([''] + [''] + [''] + [", ".join(["%i" % i for i in c])
                                         if len(c) > 0 and len(c) < n_clfs-2 else ("all" if len(c) == n_clfs-2 else "---")
                                         for c in conclusions])
        
# Wilcoxon
# DATASETS x MODALITIES x CLF
ranks = np.array(ranks)
# print(ranks, ranks.shape)
mean_ranks = np.mean(ranks, axis=0)

w_statistic = np.zeros((n_clfs-1, n_clfs-1))
p = np.zeros((n_clfs-1, n_clfs-1))

for i in range(n_clfs-1):
    for j in range(n_clfs-1):
        w_statistic[i, j], p[i, j] = ranksums(ranks.T[i], ranks.T[j])

_ = np.where((p < alfa) * (w_statistic > 0))
conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]

all.append(['Average rank'] + [''] + [''] + ["%.3f" % v for v in mean_ranks])
all.append([''] + [''] + [''] + [", ".join(["%i" % i for i in c])
                             if len(c) > 0 and len(c) < n_clfs-2 else ("all" if len(c) == n_clfs-2 else "---")
                             for c in conclusions])
        
print(tabulate(all, headers=["Dataset"] + ["M"] + alg_names, floatfmt=".3f", tablefmt="latex_booktabs"))


# Multi
all = []
ranks = []
for data_id, data in enumerate(topics[10:]):
    # MODALITY x ALG x FOLDS
    data_scores = scores_all_multi[:, data_id]
    for modality_id, modality in enumerate(modalities[0]):
        # ALG x FOLDS
        modality_scores = data_scores[modality_id]
        ranks.append(rankdata(np.mean(modality_scores[1:], axis=1)))
        
        all.append(["%s"% data] + ["%s"% modality] + ["%.3f" % score for score in np.mean(modality_scores, axis=1)])
        
        # t-test corrected
        T, p = np.array([[t_test_corrected(modality_scores[[1, 2, 3]][i],
                               modality_scores[[1, 2, 3]][j]) if i != j else (0.0, 1.0)
                              for i in range(n_clfs-1)]
                             for j in range(n_clfs-1)]
                        ).swapaxes(0, 2)
        mean_adv = np.mean(modality_scores[[1, 2, 3]], axis=1) < np.mean(modality_scores[[1, 2, 3]], axis=1)[:, np.newaxis]
        stat_adv = p < alfa
        
        _ = np.where(stat_adv * mean_adv)
        conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]
        
        all.append([''] + [''] + [''] + [", ".join(["%i" % i for i in c])
                                         if len(c) > 0 and len(c) < n_clfs-2 else ("all" if len(c) == n_clfs-2 else "---")
                                         for c in conclusions])
        
# Wilcoxon
# DATASETS x MODALITIES x CLF
ranks = np.array(ranks)
# print(ranks, ranks.shape)
mean_ranks = np.mean(ranks, axis=0)

w_statistic = np.zeros((n_clfs-1, n_clfs-1))
p = np.zeros((n_clfs-1, n_clfs-1))

for i in range(n_clfs-1):
    for j in range(n_clfs-1):
        w_statistic[i, j], p[i, j] = ranksums(ranks.T[i], ranks.T[j])

_ = np.where((p < alfa) * (w_statistic > 0))
conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]

all.append(['Average rank'] + [''] + [''] + ["%.3f" % v for v in mean_ranks])
all.append([''] + [''] + [''] + [", ".join(["%i" % i for i in c])
                             if len(c) > 0 and len(c) < n_clfs-2 else ("all" if len(c) == n_clfs-2 else "---")
                             for c in conclusions])
        
print(tabulate(all, headers=["Dataset"] + ["M"] + alg_names, floatfmt=".3f", tablefmt="latex_booktabs"))