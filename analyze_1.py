import numpy as np
from tabulate import tabulate
from utils import t_test_corrected
from scipy.stats import rankdata, ranksums


# DATASETS x MODALITIES x FOLDS x CLF
scores = np.load("scores/experiment_1_52.npy")
scores_binary = scores[:10]
scores_multi = scores[10:]

topics = ["HR", "DS", "HM", "BW", "AM", "FS", "FW", "MS", "FM", "AT", 
          "HMSWW", "FMSW", "FMNSSW", "FFW", "CDFS", "AFHMS", "ACMD", "ACCDDHMRS", "ABHMW", "ABFHSS"]

clfs = [
    "GNB", 
    "LogR", 
    "CART"]
n_clfs = len(clfs)

modalities = [
    ["img", "txt"],
]

alfa = .05

# Binary
all = []
ranks = []
for data_id, data in enumerate(topics[:10]):
    # MODALITIES x FOLDS x CLF
    data_scores = scores_binary[data_id]
    for modality_id, modality in enumerate(modalities[0]):
        # FOLDS x CLF
        modality_scores = data_scores[modality_id]
        
        ranks.append(rankdata(np.mean(modality_scores, axis=0)))
        
        all.append(["%s"% data] + ["%s"% modality] + ["%.3f" % score for score in np.mean(modality_scores, axis=0)])
        
        # t-test corrected
        T, p = np.array([[t_test_corrected(modality_scores[:, i],
                               modality_scores[:, j]) if i != j else (0.0, 1.0)
                              for i in range(n_clfs)]
                             for j in range(n_clfs)]
                        ).swapaxes(0, 2)
        mean_adv = np.mean(modality_scores, axis=0) < np.mean(modality_scores, axis=0)[:, np.newaxis]
        stat_adv = p < alfa
        
        _ = np.where(stat_adv * mean_adv)
        conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]
        
        all.append([''] + [''] + [", ".join(["%i" % i for i in c])
                                         if len(c) > 0 and len(c) < n_clfs-1 else ("all" if len(c) == n_clfs-1 else "---")
                                         for c in conclusions])
        
# Wilcoxon
# DATASETS x MODALITIES x CLF
ranks = np.array(ranks)
# print(ranks, ranks.shape)
mean_ranks = np.mean(ranks, axis=0)

w_statistic = np.zeros((n_clfs, n_clfs))
p = np.zeros((n_clfs, n_clfs))

for i in range(n_clfs):
    for j in range(n_clfs):
        w_statistic[i, j], p[i, j] = ranksums(ranks.T[i], ranks.T[j])

_ = np.where((p < alfa) * (w_statistic > 0))
conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]

all.append(['Average rank'] + [''] + ["%.3f" % v for v in mean_ranks])
all.append([''] + [''] + [", ".join(["%i" % i for i in c])
                             if len(c) > 0 and len(c) < len(clfs)-1 else ("all" if len(c) == len(clfs)-1 else "---")
                             for c in conclusions])
        
print(tabulate(all, headers=["Dataset"] + ["Modality"] + clfs, floatfmt=".3f"))

# Multiclass
all = []
ranks = []
for data_id, data in enumerate(topics[10:]):
    # print(data)
    data_scores = scores_multi[data_id]
    for modality_id, modality in enumerate(modalities[0]):
        # print(modality)
        modality_scores = data_scores[modality_id]
        
        ranks.append(rankdata(np.mean(modality_scores, axis=0)))
        
        all.append(["%s"% data] + ["%s"% modality] + ["%.3f" % score for score in np.mean(modality_scores, axis=0)])
        
        T, p = np.array([[t_test_corrected(modality_scores[:, i],
                               modality_scores[:, j]) if i != j else (0.0, 1.0)
                              for i in range(n_clfs)]
                             for j in range(n_clfs)]
                        ).swapaxes(0, 2)
        mean_adv = np.mean(modality_scores, axis=0) < np.mean(modality_scores, axis=0)[:, np.newaxis]
        stat_adv = p < alfa
        
        _ = np.where(stat_adv * mean_adv)
        conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]
        
        all.append([''] + [''] + [", ".join(["%i" % i for i in c])
                                         if len(c) > 0 and len(c) < n_clfs-1 else ("all" if len(c) == n_clfs-1 else "---")
                                         for c in conclusions])
        

# Wilcoxon
# DATASETS x MODALITIES x CLF
ranks = np.array(ranks)
# print(ranks, ranks.shape)
mean_ranks = np.mean(ranks, axis=0)

w_statistic = np.zeros((n_clfs, n_clfs))
p = np.zeros((n_clfs, n_clfs))

for i in range(n_clfs):
    for j in range(n_clfs):
        w_statistic[i, j], p[i, j] = ranksums(ranks.T[i], ranks.T[j])

_ = np.where((p < alfa) * (w_statistic > 0))
conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]

all.append(['Average rank'] + [''] + ["%.3f" % v for v in mean_ranks])
all.append([''] + [''] + [", ".join(["%i" % i for i in c])
                             if len(c) > 0 and len(c) < len(clfs)-1 else ("all" if len(c) == len(clfs)-1 else "---")
                             for c in conclusions])

print(tabulate(all, headers=["Dataset"] + ["Modality"] + clfs, floatfmt=".3f"))
            
            