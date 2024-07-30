"""

"""
import numpy as np
from tabulate import tabulate
from utils import t_test_corrected, cv52cft
from scipy.stats import rankdata, ranksums, wilcoxon, ttest_rel

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
modalities_tab = [
    ["Visual", "Text"],
]

# alg_names = [
#     "full",
#     "early fusion",
#     "late fusion",
#     "seed", 
#     "single",
#     "cross",
#     ]

alg_names = [
    "FULL",
    "EF",
    "LF",
    "PRE", 
    "UNI",
    "CMCSL",
    ]

n_clfs = len(alg_names)

# TOPIC x ALG x TIMES x FOLDS
scores_m1 = np.load("scores/experiment_3_m1_52.npy")
scores_m2 = np.load("scores/experiment_3_m2_52.npy")
scores_m1_mc = np.load("scores/experiment_3_m1_52_mc.npy")
scores_m2_mc = np.load("scores/experiment_3_m2_52_mc.npy")
# MODALITY x TOPIC x ALG x TIMES x FOLDS x CLFS
scores_all = np.stack((scores_m1, scores_m2))
scores_all_mc = np.stack((scores_m1_mc, scores_m2_mc))
# All three base classifiers
scores_all = np.concatenate((scores_all.reshape(2, 20, 4, 20, 10, 1), scores_all_mc), axis=5)

# Only GNB – last dim
base_clf = 0
# MODALITY x TOPIC x ALG x FOLDS
scores_all = np.mean(scores_all, axis=3)
scores_all = scores_all[:, :, :, :, base_clf]
scores_all_binary = scores_all[:, :10]
scores_all_multi = scores_all[:, 10:]

"""
LATE FUSION
"""
# TOPIC x ALG X TIMES x FOLDS x BASE
scores_lf = np.load("scores/experiment_3_52_late_fusion_all_clfs.npy")
# TOPIC x ALG X TIMES x FOLDS
scores_lf_gnb = scores_lf[:, :, :, :, 0]
# TOPIC x ALG x FOLDS
scores_lf_gnb = np.mean(scores_lf_gnb, axis=2)
scores_lf_gnb_binary = np.array([scores_lf_gnb[:10], scores_lf_gnb[:10]])
scores_lf_gnb_multi = np.array([scores_lf_gnb[10:], scores_lf_gnb[10:]])

scores_all_binary = np.concatenate((scores_all_binary, scores_lf_gnb_binary), axis=2)
scores_all_multi = np.concatenate((scores_all_multi, scores_lf_gnb_multi), axis=2)

# # MEthod order
# scores_all_binary = scores_all_binary[:, :, [0, 4, 1, 2, 3], :]
# scores_all_multi = scores_all_multi[:, :, [0, 4, 1, 2, 3], :]

"""
EARLY FUSION
"""
# TOPIC x ALG X TIMES x FOLDS x BASE
scores_ef = np.load("scores/experiment_3_52_early_fusion_all_clfs.npy")
# TOPIC x ALG X FOLDS
scores_ef_gnb = scores_ef[:, :, 0, :, 0]
# TOPIC x ALG x FOLDS
scores_ef_gnb_binary = np.array([scores_ef_gnb[:10], scores_ef_gnb[:10]])
scores_ef_gnb_multi = np.array([scores_ef_gnb[10:], scores_ef_gnb[10:]])

scores_all_binary = np.concatenate((scores_all_binary, scores_ef_gnb_binary), axis=2)
scores_all_multi = np.concatenate((scores_all_multi, scores_ef_gnb_multi), axis=2)

# MEthod order
scores_all_binary = scores_all_binary[:, :, [0, 5, 4, 1, 2, 3], :]
scores_all_multi = scores_all_multi[:, :, [0, 5, 4, 1, 2, 3], :]

# Binary
all = []
ranks_ = []
wilc = []
for data_id, data in enumerate(topics[:10]):
    # MODALITY x ALG x FOLDS
    data_scores = scores_all_binary[:, data_id]
    for modality_id, modality in enumerate(modalities_tab[0]):
        # ALG x FOLDS
        modality_scores = data_scores[modality_id]
        ranks_.append(rankdata(np.mean(modality_scores[3:], axis=1)))
        wilc.append(np.mean(modality_scores[3:], axis=1))
        all.append(["%s"% data] + ["%s"% modality] + ["%.3f" % score for score in np.mean(modality_scores, axis=1)])
        
        # t-test corrected
        print(modality_scores.shape)
        T, p = np.array([[cv52cft(modality_scores[[3, 4, 5]][i],
                               modality_scores[[3, 4, 5]][j]) if i != j else (0.0, 1.0)
                              for i in range(n_clfs-3)]
                             for j in range(n_clfs-3)]
                        ).swapaxes(0, 2)
        mean_adv = np.mean(modality_scores[[3, 4, 5]], axis=1) < np.mean(modality_scores[[3, 4, 5]], axis=1)[:, np.newaxis]
        stat_adv = p < alfa
        
        _ = np.where(stat_adv * mean_adv)
        conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs-1)]
        
        all.append([''] +[''] +[''] + [''] + [''] + [", ".join(["%i" % i for i in c])
                                         if len(c) > 0 and len(c) < n_clfs-4 else ("all" if len(c) == n_clfs-4 else "---")
                                         for c in conclusions])
        
# Wilcoxon
# DATASETS x MODALITIES x CLF
ranks = np.array(wilc)
# print(ranks, ranks.shape)
mean_ranks = np.mean(ranks_, axis=0)

# ranks = np.array(wilc)


w_statistic = np.zeros((n_clfs-3, n_clfs-3))
p = np.zeros((n_clfs-3, n_clfs-3))

for i in range(n_clfs-3):
    for j in range(n_clfs-3):
        w_statistic[i, j], p[i, j] = wilcoxon(ranks.T[i], ranks.T[j], zero_method="zsplit", alternative="greater")

_ = np.where((p < alfa) * (w_statistic > 0))
conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs-1)]

all.append(['Average rank'] + [''] + [''] + [''] + [''] + ["%.3f" % v for v in mean_ranks])
all.append([''] +[''] +[''] + [''] + [''] + [", ".join(["%i" % i for i in c])
                             if len(c) > 0 and len(c) < n_clfs-4 else ("all" if len(c) == n_clfs-4 else "---")
                             for c in conclusions])
        
print(tabulate(all, headers=["Dataset"] + ["M"] + alg_names, floatfmt=".3f", tablefmt="latex_booktabs"))
# exit()



# Multi
all = []
ranks_ = []
for data_id, data in enumerate(topics[10:]):
    # MODALITY x ALG x FOLDS
    data_scores = scores_all_multi[:, data_id]
    for modality_id, modality in enumerate(modalities_tab[0]):
        # ALG x FOLDS
        modality_scores = data_scores[modality_id]
        ranks_.append(rankdata(np.mean(modality_scores[3:], axis=1)))
        
        all.append(["%s"% data] + ["%s"% modality] + ["%.3f" % score for score in np.mean(modality_scores, axis=1)])
        
        # t-test corrected
        T, p = np.array([[cv52cft(modality_scores[[3, 4, 5]][i],
                               modality_scores[[3, 4, 5]][j]) if i != j else (0.0, 1.0)
                              for i in range(n_clfs-3)]
                             for j in range(n_clfs-3)]
                        ).swapaxes(0, 2)
        mean_adv = np.mean(modality_scores[[3, 4, 5]], axis=1) < np.mean(modality_scores[[3, 4, 5]], axis=1)[:, np.newaxis]
        stat_adv = p < alfa
        
        _ = np.where(stat_adv * mean_adv)
        conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs-1)]
        
        all.append(['']+ [''] +[''] + [''] + [''] + [", ".join(["%i" % i for i in c])
                                         if len(c) > 0 and len(c) < n_clfs-4 else ("all" if len(c) == n_clfs-4 else "---")
                                         for c in conclusions])
        
# Wilcoxon
# DATASETS x MODALITIES x CLF
ranks = np.array(wilc)
# print(ranks, ranks.shape)
mean_ranks = np.mean(ranks_, axis=0)

w_statistic = np.zeros((n_clfs-3, n_clfs-3))
p = np.zeros((n_clfs-3, n_clfs-3))

for i in range(n_clfs-3):
    for j in range(n_clfs-3):
        w_statistic[i, j], p[i, j] = wilcoxon(ranks.T[i], ranks.T[j], zero_method="zsplit", alternative="greater")

_ = np.where((p < alfa) * (w_statistic > 0))
conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs-1)]

all.append(['Average rank']+ [''] +[''] + [''] + [''] + ["%.3f" % v for v in mean_ranks])
all.append(['']+ [''] +[''] + [''] + [''] + [", ".join(["%i" % i for i in c])
                             if len(c) > 0 and len(c) < n_clfs-4 else ("all" if len(c) == n_clfs-4 else "---")
                             for c in conclusions])
        
print(tabulate(all, headers=["Dataset"] + ["M"] + alg_names, floatfmt=".3f", tablefmt="latex_booktabs"))