import os
from matplotlib import pyplot as plt
import numpy as np
from pandia import RESULTS_PATH
from pandia.analysis.stream_illustrator import DPI, FIG_EXTENSION, setup_diagrams_path
from pandia.constants import M


def main():
    bw = np.arange(1, 5)
    reward_drl = [-1.72, 2.85, 4.32, 5.83]
    reward_gcc = [-2.84, 2.01, 3.98, 4.99]
    plt.close()
    path = os.path.join(RESULTS_PATH, 'sb3_vs_gcc')
    setup_diagrams_path(path)
    fig, ax = plt.subplots(figsize=(4, 2))
    width = .3
    ax.bar(bw - width / 2, reward_gcc, width, label='GCC') # type: ignore
    ax.bar(bw + width / 2, reward_drl, width, label='DRL') # type: ignore
    ax.set_xticks(bw)
    plt.xlabel('Bandwidth (Mbps)')
    plt.ylabel('Reward (x100)')
    plt.legend()
    plt.tight_layout(pad=.1)
    plt.savefig(os.path.join(path, f'reward.{FIG_EXTENSION}'), dpi=DPI)



if __name__ == "__main__":
    main()
