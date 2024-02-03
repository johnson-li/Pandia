import json
import os

from matplotlib import pyplot as plt

from pandia import RESOURCE_PATH, RESULTS_PATH
from pandia.analysis.stream_illustrator import DPI, FIG_EXTENSION, setup_diagrams_path


def main():
    path = os.path.join(RESOURCE_PATH, "tensorboard")
    path = os.path.join(path, 'direct_training_value_loss.json')
    data = json.load(open(path))[:160]
    steps = [d[1] for d in data]
    losses = [d[2] for d in data]

    fig_path = os.path.join(RESULTS_PATH, "analysis_value_loss")
    setup_diagrams_path(fig_path)
    plt.close()
    fig = plt.gcf()
    fig.set_size_inches(3, 1.5)
    plt.plot(steps, losses)
    plt.xlabel('Steps')
    plt.ylabel('Value loss')
    plt.tight_layout(pad=.1)
    plt.savefig(os.path.join(fig_path, f'value_loss.{FIG_EXTENSION}'), dpi=DPI)


if __name__ == "__main__":
    main()
