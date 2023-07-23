import os
from pandia import DIAGRAMS_PATH
from pandia.eval.eval_static import run


def main():
    for bitrate in [300, 500, 800, 1000, 1300, 1500, 1800, 2000, 2300, 2500, 2800, 3000]:
        prefix = f'eval_bitrate_{bitrate}'
        sender_path = os.path.join(DIAGRAMS_PATH, prefix, 'eval_sender_log.txt')
        if os.path.exists(sender_path) and os.path.getsize(sender_path) > 1024 * 1024:
            print(f'Skipping {prefix}')
            continue
        run(bw=3 * 1024, delay=10, loss=0, bitrate=bitrate, 
            working_dir=os.path.join(DIAGRAMS_PATH, 'eval_multi_setup', prefix))


def analyse():
    for bitrate in [300, 500, 800, 1000, 1300, 1500, 1800, 2000, 2300, 2500, 2800, 3000]:
        prefix = f'eval_bitrate_{bitrate}'
        sender_path = os.path.join(DIAGRAMS_PATH, prefix, 'eval_sender_log.txt')


if __name__ == "__main__":
    main()
