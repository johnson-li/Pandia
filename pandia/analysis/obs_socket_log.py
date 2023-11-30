import os
from struct import unpack
from pandia import RESULTS_PATH
from pandia.agent.observation_thread import ObservationThread
from pandia.analysis.stream_illustrator import illustrate_frame, illustrate_frame_ts
from pandia.log_analyzer_sender import StreamingContext


def main():
    obs = ObservationThread(None)
    context = StreamingContext()
    obs.context = context
    with open('/tmp/obs.log', 'rb') as f:
        while True:
            head = f.read(16)
            if head:
                msg_size, msg_type = unpack('QQ', head)
                msg = f.read(msg_size - 16)
                obs.parse_data(msg, msg_type)
            else:
                break

    path = os.path.join(RESULTS_PATH, 'obs_socket_log')
    illustrate_frame(path, context)


if __name__ == '__main__':
    main()
