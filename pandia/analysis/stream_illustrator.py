import os
from matplotlib import pyplot as plt
import numpy as np
from pandia.constants import K
from pandia.log_analyzer_sender import StreamingContext


def illustrate_frame_ts(path: str, context: StreamingContext):
    if not context.frames:
        return
    ids = list(sorted(context.frames.keys()))
    ts_min = context.frames[ids[0]].captured_at_utc
    frames = []
    for frame_id in ids:
        frame = context.frames[frame_id]
        if frame.decoding_at:
            frames.append((frame.captured_at_utc, 
                           frame.encoded_at - frame.captured_at,
                           frame.assembled_at_utc - frame.captured_at_utc,
                           frame.decoded_at_utc - frame.captured_at_utc))
    def plot(i, j):
        x = np.array([f[i] for f in frames])
        y = np.array([f[j] for f in frames])
        indexes = y > 0
        plt.plot(x[indexes] - ts_min, y[indexes] * 1000)

    plt.close()
    for i in range(1, 4):
        plot(0, i)
    plt.ylim(0, 50)
    plt.xlabel('Frame capture time (s)')
    plt.ylabel('Delay (ms)')
    plt.legend(['Encoding Delay', 'Assembly Delay', 'Decoding Delay'])
    plt.savefig(os.path.join(path, 'frame-ts.pdf'))

def illustrate_frame_spec(path: str, context: StreamingContext):
    if not context.frames:
        return
    ids = list(sorted(context.frames.keys()))
    ts_min = context.frames[ids[0]].captured_at_utc
    encoded_size_data = []
    resolution_data = []
    for frame_id in ids:
        frame = context.frames[frame_id]
        if frame.encoded_size > 0:
            encoded_size_data.append((frame.captured_at_utc, frame.encoded_size))
        resolution_data.append((frame.captured_at_utc, frame.height))
    encoded_size_data = np.array(encoded_size_data)
    resolution_data = np.array(resolution_data)
    plt.close()
    fig, ax1 = plt.subplots()
    ax1.plot(encoded_size_data[:, 0] - ts_min, encoded_size_data[:, 1] / K, 'b')
    ax1.set_xlabel('Frame capture time (s)')
    ax1.set_ylabel('Encoded size (KB)')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    indexes = resolution_data[:, 1] > 0
    ax2.plot(resolution_data[indexes, 0] - ts_min, resolution_data[indexes, 1], 'r')
    ax2.set_ylabel('Resolution (height)')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'frame-spec.pdf'))

def illustrate_frame_bitrate(path: str, context: StreamingContext):
    if not context.frames:
        return
    ids = list(sorted(context.frames.keys()))
    ts_min = context.frames[ids[0]].captured_at_utc
    bitrate_data = []
    for frame_id in ids:
        frame = context.frames[frame_id]
        if frame.bitrate > 0:
            bitrate_data.append((frame.captured_at_utc, frame.bitrate))
    plt.close()
    plt.plot([f[0] - ts_min for f in bitrate_data], [f[1] / K for f in bitrate_data])
    plt.xlabel('Frame capture time (s)')
    plt.ylabel('Bitrate (Kbps)')
    plt.savefig(os.path.join(path, 'frame-bitrate.pdf'))

def illustrate_frame(path, context):
    os.makedirs(path, exist_ok=True)
    illustrate_frame_ts(path, context)
    illustrate_frame_spec(path, context)
    illustrate_frame_bitrate(path, context)
