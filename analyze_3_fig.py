"""

"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib


matplotlib.rcParams.update({'font.size': 13, "font.family" : "monospace"})


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

modalities_names = [
    # ["video", "audio", "y"],
    ["VISUAL", "TEXT"],
]

alg_names = [
    "FULL", 
    "PRE", 
    "UNI",
    "CMCSL",
    ]

# ls = [":", "--", "-.", "-"]
ls = ["-", ":", "--", (10, (20, 40))]
# ls = ["-", ":", "--", "-."]
# colors = ["black", "dodgerblue", "blue", "red"]

n_times = np.array([i+1 for i in range(20)])

# TOPIC x ALG x TIMES x FOLDS
scores_m1 = np.load("scores/experiment_3_m1_52.npy")
scores_m2 = np.load("scores/experiment_3_m2_52.npy")
# TOPIC x ALG x TIMES
scores_m1 = np.mean(scores_m1, axis=3)
scores_m2 = np.mean(scores_m2, axis=3)


"""
Additional late fusion
"""
# DATASETS x ALG X TIMES x FOLDS x BASE
scores_lf = np.load("scores/experiment_3_52_late_fusion_all_clfs.npy")
# DATASETS x ALG X TIMES x BASE
scores_lf = np.mean(scores_lf, axis=3)
print(scores_lf.shape)
print(scores_lf[0, 0, :, 0])
# DATASETS X TIMES
scores_lf_gnb = scores_lf[:, 0, :, 0]
print(scores_lf_gnb.shape)

"""
"""

"""
Additional early fusion
"""
# DATASETS x ALG X TIMES x FOLDS x BASE
scores_ef = np.load("scores/experiment_3_52_early_fusion_all_clfs.npy")
# DATASETS x ALG X TIMES x BASE
scores_ef = np.mean(scores_ef, axis=3)
print(scores_ef.shape)
print(scores_ef[0, 0, :, 0])
# DATASETS X TIMES
scores_ef_gnb = scores_ef[:, 0, :, 0]
scores_ef_gnb = np.repeat(scores_ef_gnb, 20, 1)
print(scores_ef_gnb.shape)
"""
"""


lw = 1.5

# For each modality get clusters and distances
for topic_id, topic in tqdm(enumerate(topics[0]), total = 20):
    # ALG x TIMES
    topic_scores_m1 = scores_m1[topic_id]
    topic_scores_m2 = scores_m2[topic_id]
    
    topic_scores_lf = scores_lf_gnb[topic_id]
    topic_scores_ef = scores_ef_gnb[topic_id]
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.set_yticks(np.arange(0.1, 1.1, .1))
    # PP-RAI 2024
    ax.set_ylim(0.5, 1.0)
    # ax.set_ylim(0.1, 1.0)
    ax.set_xticks([i for i in n_times])
    ax.set_xlim(n_times[0], n_times[-1])
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("#samples for each class")
    ax.set_ylabel("Balanced accuracy")
    
    for algorithm_id in range(len(alg_names)):
        alg_scores_m1 = topic_scores_m1[algorithm_id]
        alg_scores_m2 = topic_scores_m2[algorithm_id]
        
        
        
        # ax.plot(n_times, alg_scores_m1, c=colors[algorithm_id], ls="-", label = "%s %s %.3f" % (modalities[0][0], alg_names[algorithm_id], np.mean(alg_scores_m1)))
        # ax.plot(n_times, alg_scores_m2, c=colors[algorithm_id], ls="--", label = "%s %s %.3f" % (modalities[0][1], alg_names[algorithm_id], np.mean(alg_scores_m2)))
        
        # """
        if algorithm_id == len(alg_names)-1:
            # ax.plot(n_times, alg_scores_m1, c="red", ls="-", lw=lw)
            # ax.plot(n_times, alg_scores_m2, c="blue", ls="-", lw=lw)
            
            # ax.plot(n_times, alg_scores_m1, c="blue", ls=ls[algorithm_id], lw=lw, label = "%s %s %.3f" % (modalities[0][0], alg_names[algorithm_id], np.mean(alg_scores_m1)))
            # ax.plot(n_times, alg_scores_m2, c="red", ls=ls[algorithm_id], lw=lw, label = "%s %s %.3f" % (modalities[0][1], alg_names[algorithm_id], np.mean(alg_scores_m2)))
            
            ax.plot(n_times, alg_scores_m1, c="red", ls="-.", lw=lw, label = "%s %s %.3f" % (modalities_names[0][0], alg_names[algorithm_id], np.mean(alg_scores_m1)))
            ax.plot(n_times, alg_scores_m2, c="blue", ls="-.", lw=lw, label = "%s %s %.3f" % (modalities_names[0][1], alg_names[algorithm_id], np.mean(alg_scores_m2)))
            
        else:
            ax.plot(n_times, alg_scores_m1, c="red", lw=.8, ls=ls[algorithm_id], label = "%s %s %.3f" % (modalities_names[0][0], alg_names[algorithm_id], np.mean(alg_scores_m1)))
            ax.plot(n_times, alg_scores_m2, c="blue", lw=.8, ls=ls[algorithm_id], label = "%s %s %.3f" % (modalities_names[0][1], alg_names[algorithm_id], np.mean(alg_scores_m2)))
            
    ax.plot(n_times, topic_scores_lf, c="black", ls="-", lw=1, label = "LATE FUSION %.3f" % (np.mean(topic_scores_lf)))
    ax.plot(n_times, topic_scores_ef, c="black", ls="--", lw=1, label = "EARLY FUSION %.3f" % (np.mean(topic_scores_ef)))
    print(topic_scores_lf)
    
        # """
        
    plt.grid((.7, .7, .7), ls=":")
    plt.tight_layout()
    plt.legend(frameon=False, ncol=3, loc="upper center")
    # plt.savefig("figures/ex3/ex3_%s.png" % (topic), dpi=200)
    # plt.savefig("figures/ex3/ex3_%s.eps" % (topic), dpi=200)
    plt.savefig("figures/ex3/ex3_%s_pprai24.png" % (topic), dpi=200)
    plt.savefig("figures/ex3/ex3_%s_pprai24.eps" % (topic), dpi=200)
    plt.close()
        
        