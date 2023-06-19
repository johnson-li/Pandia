import os

from pandia import DIAGRAMS_PATH
from pandia.log_analyzer import StreamingContext, analyze_stream, parse_line
from pandia.log_analyzer_receiver import Stream, parse_line as parse_line_receiver, analyze as analyze_receiver


def analyse_sender(prefix) -> None:
    ctx = StreamingContext()
    exp_dir = os.path.join(DIAGRAMS_PATH, prefix)
    sender_log = os.path.join(exp_dir, 'eval_sender_log.txt')
    for line in open(sender_log).readlines():
        parse_line(line, ctx)
    analyze_stream(ctx, exp_dir)


def analyse_receiver(prefix) -> None:
    ctx = Stream()
    exp_dir = os.path.join(DIAGRAMS_PATH, prefix)
    receiver_log = os.path.join(exp_dir, 'eval_receiver_log.txt')
    for line in open(receiver_log).readlines():
        line = line.strip()
        if line:
            parse_line_receiver(line, ctx)
    analyze_receiver(ctx)


def run(prefix):
    print(f'Processing {prefix}')
    analyse_sender(prefix)
    analyse_receiver(prefix)


if __name__ == "__main__":
    for width in [360, 720, 1080]:
        for fps in [5, 10, 30]:
            for bitrate in [512, 1024, 2048, 4096]:
                run(f'eval_{width}p_{fps}_{bitrate}kbps')
