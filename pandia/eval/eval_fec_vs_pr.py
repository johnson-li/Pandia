import os

from matplotlib import pyplot as plt
from pandia import DIAGRAMS_PATH, RESULTS_PATH
from pandia.eval.eval_static import run_exp
from pandia.log_analyzer_hybrid import count_rtp_distribution, main as analyzer_main
from pandia.log_analyzer_receiver import get_stream
from pandia.log_analyzer_sender import get_stream_context


RESULT_DIR = os.path.join(RESULTS_PATH, "fec_vs_pr")
FEC_RATIO = [5, 10, 50, 100]
PR = [20, 50, 100]

def run():
    for fec_ratio in FEC_RATIO:
        for pr in PR:
            action = {
                'pacing_rate': pr,
                'bitrate': 10 * 1024,
                'fec_key': int(255 * fec_ratio / 100),
                'fec_delta': int(255 * fec_ratio / 100),
            }
            for i in range(3):
                try:
                    print(f'Evaluating with action: {action}')
                    result_dir = os.path.join(RESULT_DIR, f'fec_{fec_ratio}_pr_{pr}')
                    run_exp(action=action, result_dir=result_dir, duration=60)
                    analyzer_main(result_dir)
                    break
                except Exception as e:
                    print(e)
                    continue


def draw_diagram():
    data = {}
    for fec_ratio in FEC_RATIO:
        data.setdefault(fec_ratio, {})
        for pr in PR:
            result_dir = os.path.join(RESULT_DIR, f'fec_{fec_ratio}_pr_{pr}')
            if os.path.isdir(result_dir):
                try: 
                    stream = get_stream(result_dir)
                    stream_context = get_stream_context(result_dir)
                    recv_pkts, recovery_pkts, retrans_pkts = count_rtp_distribution(stream_context, stream)
                    all_pkts = recv_pkts + recovery_pkts + retrans_pkts
                    data[fec_ratio][pr] = recovery_pkts / (recovery_pkts + retrans_pkts)
                    print(f'fec_ratio: {fec_ratio}, pr: {pr}, '
                        f'recovery_ratio: {recovery_pkts / all_pkts} ({recovery_pkts / (recovery_pkts + retrans_pkts)})')
                except Exception as e:
                    print(e)
                    continue
    plt.close();
    legend = []
    for fec_ratio in FEC_RATIO:
        legend.append(f'fec_ratio: {fec_ratio}%')
        pr_list = sorted(data[fec_ratio].keys())
        plt.plot(pr_list, [data[fec_ratio][pr] for pr in pr_list])
    plt.legend(legend)
    plt.xlabel('Pacing rate (kbos)')
    plt.ylabel('RTP Recovery ratio (%)')
    plt.savefig(os.path.join(DIAGRAMS_PATH, 'fec_vs_pr.pdf'))


def main():
    run()
    draw_diagram()


if __name__ == "__main__":
    main()
