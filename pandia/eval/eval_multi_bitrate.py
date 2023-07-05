import os
import re

from matplotlib import pyplot as plt

from pandia import DIAGRAMS_PATH, RESULTS_PATH
from pandia.eval.eval_static import analyze, run_exp


RESULT_DIR = os.path.join(RESULTS_PATH, "eval_multi_bitrate")
BITRATES = [.2, .5, .8, 1, 2, 5, 8, 10]


def draw_diagram():
    x = []
    y = []
    for bitrate in BITRATES:
        result_dir = os.path.join(RESULT_DIR, f'bitrate_{bitrate}')
        sender_log = os.path.join(result_dir, 'eval_sender.log')
        bw = 0
        for line in open(sender_log, 'r').readlines():
            if 'SetPacingRates' in line:
                m = re.match(re.compile(
                    '.*\\[(\\d+)\\] SetPacingRates, pacing rate: (\\d+) kbps, pading rate: (\\d+) kbps.*'), line)
                bandwidth = float(m[2])
                bw = max(bw, bandwidth)
        print(f'Bitrate: {bitrate}, bandwidth: {bw}')
        x.append(bitrate)
        y.append(bw)

    plt.close()
    plt.plot(x, y)
    plt.xlabel('Encoding bitrate (kbps)')
    plt.ylabel('Estimated bandwidth (kbps)')
    plt.plot()
    plt.savefig(os.path.join(DIAGRAMS_PATH, 'multi_bitrate.pdf'))


def run():
    for bitrate in BITRATES:
        action = {
            'bitrate': int(bitrate * 1024),
        }
        print(f'Evaluating with action: {action}')
        result_dir = os.path.join(RESULT_DIR, f'bitrate_{bitrate}')
        run_exp(action=action, result_dir=result_dir, duration=10)
        analyze(result_dir)


def main():
    # run()
    draw_diagram()


if __name__ == "__main__":
    main()
