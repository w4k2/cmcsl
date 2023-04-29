"""

"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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

alg_names = [
    "full", 
    "seed", 
    "single",
    "cross",
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

lw = 1.5

# For each modality get clusters and distances
for topic_id, topic in tqdm(enumerate(topics[0]), total = 20):
    # ALG x TIMES
    topic_scores_m1 = scores_m1[topic_id]
    topic_scores_m2 = scores_m2[topic_id]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12/1.618))
    ax.set_yticks(np.arange(0.1, 1.1, .1))
    ax.set_ylim(0.1, 1.0)
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
            
            ax.plot(n_times, alg_scores_m1, c="red", ls="-.", lw=lw, label = "%s %s %.3f" % (modalities[0][0], alg_names[algorithm_id], np.mean(alg_scores_m1)))
            ax.plot(n_times, alg_scores_m2, c="blue", ls="-.", lw=lw, label = "%s %s %.3f" % (modalities[0][1], alg_names[algorithm_id], np.mean(alg_scores_m2)))
        else:
            ax.plot(n_times, alg_scores_m1, c="red", lw=.8, ls=ls[algorithm_id], label = "%s %s %.3f" % (modalities[0][0], alg_names[algorithm_id], np.mean(alg_scores_m1)))
            ax.plot(n_times, alg_scores_m2, c="blue", lw=.8, ls=ls[algorithm_id], label = "%s %s %.3f" % (modalities[0][1], alg_names[algorithm_id], np.mean(alg_scores_m2)))
    
        # """
        
    plt.grid((.7, .7, .7), ls=":")
    plt.tight_layout()
    plt.savefig("figures/ex3/ex3_%s.png" % (topic), dpi=200)
    plt.close()
        
        