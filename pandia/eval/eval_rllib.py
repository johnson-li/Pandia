import os
from pandia import DIAGRAMS_PATH
from pandia.eval.eval_multi_setup import run


def main():
    bitrate = 1024 * 2
    fps = 10
    width = 1080
    run(bitrate=bitrate, fps=fps, width=width, working_dir=os.path.join(DIAGRAMS_PATH, "eval_rllib"), 
        duration=10, delay=0)


if __name__ == "__main__":
    main()