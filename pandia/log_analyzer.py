import argparse
import os

from pandia import RESULTS_PATH
from pandia.log_analyzer_hybrid import main as main_hybrid
from pandia.log_analyzer_sender import main as main_sender
from pandia.log_analyzer_receiver import main as main_receiver


def main(result_dir=os.path.join(RESULTS_PATH, 'eval_static')):
    main_sender(result_dir)
    main_receiver(result_dir)
    main_hybrid(result_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--result_dir', type=str, default=os.path.join(RESULTS_PATH, 'eval_static'))
    args = parser.parse_args()
    main(args.result_dir)
