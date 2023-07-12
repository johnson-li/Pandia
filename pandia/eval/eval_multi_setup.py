import os
from pandia import DIAGRAMS_PATH
from pandia.eval.eval_rllib import run



if __name__ == "__main__":
    for width in [360, 720, 1080]:
        for fps in [5, 10, 30]:
            for bitrate in [512, 1024, 2048, 4096]:
                prefix = f'eval_{width}p_{fps}_{bitrate}kbps'
                sender_path = os.path.join(DIAGRAMS_PATH, prefix, 'eval_sender_log.txt')
                if os.path.exists(sender_path) and os.path.getsize(sender_path) > 1024 * 1024:
                    print(f'Skipping {prefix}')
                    continue
                run(bitrate=bitrate, fps=fps, width=width, working_dir=os.path.join(DIAGRAMS_PATH, 'eval_multi_setup', prefix))
