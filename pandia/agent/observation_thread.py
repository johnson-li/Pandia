
import os
import socket
from struct import unpack
import threading
import time
from typing import Optional
from pandia.context.frame_context import FrameContext
from pandia.context.packet_context import PacketContext
from pandia.context.streaming_context import StreamingContext

from pandia.log_analyzer_sender import PACKET_TYPES, kTimeWrapPeriod


class ObservationThread(threading.Thread):
    def __init__(self, sock, logging_path=None) -> None:
        super().__init__()
        self.sock: socket.socket = sock
        self.context: Optional[StreamingContext] = None
        self.stop_event = threading.Event()
        if logging_path:
            if os.path.exists(logging_path):
                os.remove(logging_path)
            self.logging_file = open(logging_path, 'wb+')
        else:
            self.logging_file = None

    def stop(self):
        self.stop_event.set()
        self.sock.close()

    def run(self) -> None:
        while not self.stop_event.is_set():
            data, addr = self.sock.recvfrom(1024)
            # print(f'Got {len(data)} bytes from {addr}')
            if not data:
                continue
            if self.logging_file:
                self.logging_file.write(data)
            msg_size = unpack('Q', data[:8])[0]
            if msg_size != len(data):
                print(f'ERROR: msg size {msg_size} != {len(data)}')
                continue
            msg_type = unpack('Q', data[8:16])[0]
            msg = data[16:]
            if msg_type == 0:
                print(f'WebRTC receiver is started.')
                continue
            try:
                self.parse_data(msg, msg_type)
            except Exception as e:
                data_hex = ''.join('{:02x}'.format(x) for x in msg)
                print(f'Msg type: {msg_type}, Error: {e}, Data: {len(msg)} bytes')

    def parse_data(self, data, msg_type):
        if self.context is None:
            print(f'ERROR: context is not initialized yet.')
            return
        context: StreamingContext = self.context
        if msg_type == 1:  # Frame captured
            ts, frame_id, width, height, frame_ts, frame_utc_ts = unpack('QQQQQQ', data)
            ts /= 1000
            frame_utc_ts /= 1000
            frame = FrameContext(frame_id, ts)
            frame.captured_at_utc = frame_utc_ts
            context.last_captured_frame_id = frame_id
            context.frames[frame_id] = frame
            [mb.on_frame_added(frame, ts) for mb in context.monitor_blocks.values()]
        elif msg_type == 2:  # Apply FEC rates
            pass
        elif msg_type == 3:  # Setup codec
            ts, = unpack('Q', data)
            ts /= 1000
            if context.start_ts <= 0:
                context.start_ts = ts
                print(f'Codec is setup, start ts: {ts}')
        elif msg_type == 4:  # Packet added 
            ts, rtp_id, seq_num, first_in_frame, last_in_frame, frame_id, rtp_type, \
                retrans_seq_num, allow_retrans, size = unpack('QQQQQQQQQQ', data)
            ts /= 1000
            if rtp_id > 0:
                # log(f'Packet sent: {rtp_id}')
                packet = PacketContext(rtp_id)
                packet.seq_num = seq_num
                packet.packet_type = PACKET_TYPES[rtp_type]
                packet.frame_id = frame_id
                packet.first_packet_in_frame = first_in_frame
                packet.last_packet_in_frame = last_in_frame
                packet.allow_retrans = allow_retrans
                packet.retrans_ref = retrans_seq_num
                packet.size = size
                context.packets[rtp_id] = packet
                context.packet_id_map[seq_num] = rtp_id
                [mb.on_packet_added(packet, ts) for mb in context.monitor_blocks.values()]
                if rtp_type == 'rtx':
                    packet.frame_id = context.packets[context.packet_id_map[retrans_seq_num]].frame_id
                if packet.frame_id > 0 and frame_id in context.frames:
                    frame: FrameContext = context.frames[frame_id]
                    frame.rtp_packets[rtp_id] = packet
                    if rtp_type == 'rtx':
                        original_rtp_id = context.packet_id_map[retrans_seq_num]
                        frame.retrans_record.setdefault(original_rtp_id, []).append(rtp_id)
                    if rtp_type == 'video':
                        frame.sequence_range[0] = min(frame.sequence_range[0], seq_num)
                        frame.sequence_range[1] = max(frame.sequence_range[1], seq_num)
        elif msg_type == 5:  # Start encoding
            ts, frame_id, height, bitrate, key_frame, fps = unpack('QQQQQQ', data)
            ts /= 1000
            bitrate *= 1024
            context.action_context.resolution = height 
            if context.action_context.bitrate <= 0:
                context.action_context.bitrate = bitrate
            if frame_id in context.frames:
                frame: FrameContext = context.frames[frame_id]
                frame.bitrate = bitrate
                frame.fps = fps
                frame.height = height
                frame.encoding_at = ts
                [mb.on_frame_encoding(frame, ts) for mb in context.monitor_blocks.values()]
            # log(f'Frame encoding started: {frame_id}, ts: {ts}')
        elif msg_type == 6:  # Finish encoding
            ts, frame_id, height, frame_size, is_key, qp = unpack('QQQQQQ', data)
            ts /= 1000
            if frame_id in context.frames:
                frame = context.frames[frame_id]
                frame.encoded_at = ts
                frame.encoded_shape = (0, height)
                frame.encoded_size = frame_size
                frame.qp = qp
                frame.is_key_frame = is_key != 0
                [mb.on_frame_encoded(frame, ts) for mb in context.monitor_blocks.values()]
            else:
                print(f'ERROR: frame {frame_id} is not found, the last one is {context.last_captured_frame_id}')
        elif msg_type == 7:  # RTCP RTT 
            ts, rtt = unpack('QQ', data)
            ts /= 1000
            rtt /= 1000
            context.rtt_data.append((ts, rtt))
        elif msg_type == 8:  # Frame decoding ack 
            ts, rtp_sequence, received_ts_utc, decoding_ts_utc, decoded_ts_utc = unpack('QQQQQ', data)
            ts /= 1000
            received_ts_utc /= 1000
            decoding_ts_utc /= 1000
            decoded_ts_utc /= 1000
            rtp_id = context.packet_id_map.get(rtp_sequence, -1)
            if rtp_id > 0 and rtp_id in context.packets:
                frame_id = context.packets[rtp_id].frame_id
                if frame_id in context.frames:
                    frame: FrameContext = context.frames[frame_id]
                    frame.assembled_at_utc = received_ts_utc
                    frame.decoding_at_utc = decoding_ts_utc
                    frame.decoded_at_utc = decoded_ts_utc
                    frame.decoded_at = ts
                    frame.decoding_at = ts - (decoded_ts_utc - decoding_ts_utc) / 1000
                    frame.assembled0_at = ts - (decoded_ts_utc - received_ts_utc) / 1000
                    context.last_decoded_frame_id = \
                            max(frame.frame_id, context.last_decoded_frame_id)
                    [mb.on_frame_decoding_updated(frame, ts) for mb in context.monitor_blocks.values()]
                    # log(f'Frame decoded: {frame_id}')
        elif msg_type == 9:  # Send video
            pass
        elif msg_type == 10:  # Apply bitrate 
            ts, bitrate, pacing_rate = unpack('QQQ', data)
            ts /= 1000
            bitrate *= 1024
            pacing_rate *= 1024
            context.drl_bitrate.append((ts, bitrate))
            context.drl_pacing_rate.append((ts, pacing_rate))
        elif msg_type == 11:  # Apply pacing rate
            ts, pacing_rate, padding_rate = unpack('QQQ', data)
            ts /= 1000
            pacing_rate *= 1024
            padding_rate *= 1024
            context.action_context.pacing_rate = pacing_rate
            context.networking.pacing_rate_data.append([ts, pacing_rate, padding_rate])
            [mb.on_pacing_rate_set(ts, pacing_rate) for mb in context.monitor_blocks.values()]
        elif msg_type == 12:  # RTCP feedback
            ts, rtp_id, lost, received_at = unpack('QQQQ', data)
            # The recv time is wrapped by kTimeWrapPeriod.
            # The fixed value 1570 should be calculated according to the current time.
            offset = int(time.time() * 1000 / kTimeWrapPeriod) - 1
            received_at = (int(received_at) + offset * kTimeWrapPeriod) / 1000
            rtp_id = int(rtp_id)
            if rtp_id in context.packets:
                packet = context.packets[rtp_id]
                packet.received_at_utc = received_at
                packet.received = lost != 1
                packet.acked_at = ts
                context.last_acked_packet_id = \
                    max(rtp_id, context.last_acked_packet_id)
                [mb.on_packet_acked(packet, ts) for mb in context.monitor_blocks.values()]
                if packet.received:
                    pkt_pre = packet
        elif msg_type == 13:  # Packet sent
            ts, rtp_id, payload_type, size, utc = unpack('QqQQQ', data)
            ts /= 1000
            utc /= 1000
            if rtp_id > 0:
                if rtp_id not in context.packets:
                    print(f'ERROR: packet {rtp_id} is not found, the last one is {context.last_egress_packet_id}')
                else:
                    packet: PacketContext = context.packets[rtp_id]
                    packet.payload_type = payload_type
                    packet.size = size
                    packet.sent_at = ts 
                    packet.sent_at_utc = utc
                    context.last_egress_packet_id = max(rtp_id, context.last_egress_packet_id)
                    [mb.on_packet_sent(packet, context.frames.get(packet.frame_id, None), ts) for mb in context.monitor_blocks.values()]
        else:
            print(f'Unknown message type: {data[0]}')


def test():
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


if __name__ == '__main__':
    test()
