"""

"""
import numpy as np
from tabulate import tabulate
from utils import t_test_corrected, cv52cft
from scipy.stats import rankdata, ranksums, wilcoxon, ttest_rel, ttest_ind
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams.update({'font.size': 18, "font.family" : "monospace"})

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
modalities_names = [
    ["Visual", "Text"],
]

alg_names = [
    # "FULL",
    # "EF",
    # "LF",
    "PRE", 
    "UNI",
    "CMCSL",
    ]
n_times = np.array([i+1 for i in range(20)])
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

# Only GNB
# MODALITY x TOPIC x ALG x TIMES x FOLDS
scores_all = scores_all[:, :, :, :, :, 0]
scores_all_binary = scores_all[:, :10]
scores_all_multi = scores_all[:, 10:]
"""
Fot tabular analysis
"""
# MODALITY x ALG x TIMES x FOLDS
scores_all_tab = np.mean(scores_all, axis=1)
scores_binary_tab = np.mean(scores_all_binary, axis=1)
scores_multi_tab = np.mean(scores_all_multi, axis=1)
# Without FULL
# MODALITY x ALG x TIMES x FOLDS
scores_all_tab = scores_all_tab[:, 1:]
scores_binary_tab = scores_binary_tab[:, 1:]
scores_multi_tab = scores_multi_tab[:, 1:]
"""
"""
# Mean over datasets and folds
# MODALITY x ALG x TIMES
scores_all = np.mean(scores_all, axis=(1, 4))
scores_all_binary = np.mean(scores_all_binary, axis=(1, 4))
scores_all_multi = np.mean(scores_all_multi, axis=(1, 4))
# Without FULL
# MODALITY x ALG x TIMES
scores_all = scores_all[:, 1:]
scores_all_binary = scores_all_binary[:, 1:]
scores_all_multi = scores_all_multi[:, 1:]

"""
PLOTS
"""

colors = ["red", "blue"]
lss = [":", "--", "-."]
lws = [.8, .8, 1.5]

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax = ax.ravel()
for mod_id, mod in enumerate(modalities_names[0]):
    for alg_id, alg in enumerate(alg_names):
        scores = scores_all_binary[mod_id, alg_id]
        ax[0].plot(n_times, scores, label="%s %s" % (mod, alg), c=colors[mod_id], ls=lss[alg_id], lw=lws[alg_id])
ax[0].grid(ls=":", c=(.7, .7, .7))
ax[0].set_xticks(n_times[::2])
ax[0].set_yticks(np.arange(0.5, 1.1, .1))
ax[0].spines[['right', 'top']].set_visible(False)
ax[0].set_xlim(1, 20)
ax[0].set_ylim(.5, 1.0)
ax[0].set_xlabel("#samples for each class")
ax[0].set_ylabel("Balanced accuracy")
ax[0].set_title("Binary datasets")
# plt.legend(frameon=False, ncol=3, loc="upper center")
# plt.tight_layout()
# plt.savefig("figures/ex3_times/binary.png", dpi=200)
# plt.savefig("figures/ex3_times/binary.eps", dpi=200)

# fig, ax = plt.subplots(1, 1, figsize=(9, 6))
for mod_id, mod in enumerate(modalities_names[0]):
    for alg_id, alg in enumerate(alg_names):
        scores = scores_all_multi[mod_id, alg_id]
        ax[1].plot(n_times, scores, label="%s %s" % (mod, alg), c=colors[mod_id], ls=lss[alg_id], lw=lws[alg_id])
ax[1].grid(ls=":", c=(.7, .7, .7))
ax[1].set_xticks(n_times[::2])
ax[1].set_yticks(np.arange(0.2, 1.1, .1))
ax[1].spines[['right', 'top']].set_visible(False)
ax[1].set_xlim(1, 20)
ax[1].set_ylim(.2, 1.0)
ax[1].set_xlabel("#samples for each class")
ax[1].set_title("Multiclass datasets")
# ax[1].set_ylabel("Balanced accuracy")
plt.tight_layout()
fig.subplots_adjust(bottom=.32)
handles, labels = ax[0].get_legend_handles_labels()
lgnd = fig.legend(handles, labels, ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), loc='lower center')
# plt.legend(frameon=False, ncol=2, loc="upper center")
plt.savefig("figures/ex3_times/whole_short.png", dpi=200)
plt.savefig("figures/ex3_times/whole_short.eps", dpi=200)

"""
STATISTICAL ANALYSIS TABLES
"""
# Binary
# MODALITY x ALG x TIMES x FOLDS
print(scores_multi_tab.shape)
# Full table with both modalities
table_both_mod = []
# For each modality
for mod_id, mod in enumerate(modalities[0]):
    # ALG x TIMES x FOLDS
    """
    scores_all_tab
    or
    scores_binary_tab
    or
    scores_multi_tab
    """
    mod_scores = scores_multi_tab[mod_id]
    """
    """
    
    table = []
    ranks_ = []
    wilc = []
    # For each number of labaled samples per class
    for times_id, times in enumerate(n_times):
        # ALG x FOLDS
        data_scores = mod_scores[:, times_id]
        ranks_.append(rankdata(np.mean(data_scores, axis=1)))
        wilc.append(np.mean(data_scores, axis=1))
        table.append(["%i"% times] + ["%.3f" % score for score in np.mean(data_scores, axis=1)])
        # table.append(["%i"% times] + ["%.3f" % score for score in np.std(data_scores, axis=1)])
        T, p = np.array([[cv52cft(data_scores[i],
                    data_scores[j]) if i != j else (0.0, 1.0)
                    for i in range(n_clfs)]
                    for j in range(n_clfs)]
                ).swapaxes(0, 2)
        mean_adv = np.mean(data_scores, axis=1) < np.mean(data_scores, axis=1)[:, np.newaxis]
        stat_adv = p < alfa
        _ = np.where(stat_adv * mean_adv)
        conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]
        table.append([''] + [", ".join(["%s" % i for i in c])
            if len(c) > 0 and len(c) < n_clfs-1 else ("all" if len(c) == n_clfs-1 else "---")
            for c in conclusions])
        
    # Wilcoxon
    # DATASETS x MODALITIES x CLF
    ranks = np.array(wilc)
    # print(ranks, ranks.shape)
    mean_ranks = np.mean(ranks_, axis=0)
    w_statistic = np.zeros((n_clfs, n_clfs))
    p = np.zeros((n_clfs, n_clfs))
    
    for i in range(n_clfs):
        for j in range(n_clfs):
            w_statistic[i, j], p[i, j] = wilcoxon(ranks.T[i], ranks.T[j], zero_method="zsplit", alternative="greater")
    _ = np.where((p < alfa) * (w_statistic > 0))
    conclusions = [list(1 + _[1][_[0] == i]) for i in range(n_clfs)]
    
    table.append(['Average rank'] + ["%.3f" % v for v in mean_ranks])
    
    table.append([''] + [", ".join(["%s" % i for i in c])
                            if len(c) > 0 and len(c) < n_clfs-1 else ("all" if len(c) == n_clfs-1 else "---")
                            for c in conclusions])
    table_both_mod.append(np.array(table))

full_table = np.concatenate((table_both_mod[0], table_both_mod[1][:, 1:]), axis=1)

print(tabulate(full_table, headers=["$b_{class}$"] + alg_names + alg_names, tablefmt="latex_booktabs"))