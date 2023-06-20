import os
from pandia import DIAGRAMS_PATH
from pandia.eval.eval_multi_setup import run


def main():
    bitrate = 1024
    fps = 10
    width = 720
    run(bitrate=bitrate, fps=fps, width=width, working_dir=os.path.join(DIAGRAMS_PATH, "eval_rllib"), duration=10)


if __name__ == "__main__":
    main()