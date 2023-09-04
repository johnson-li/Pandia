#!/bin/bash

tc qdisc del dev eth0 root 2> /dev/null || true
tc qdisc add dev eth0 root tbf rate 3000kbit burst 1000kb minburst 1540 latency 250ms
receiver=$(dig +short receiver)
stunserver=$(dig +short stun)

echo WebRTC receiver: $receiver, STUN server: $stunserver
session=pandia
tmux kill-session -t $session 2> /dev/null || true
tmux new-session -d -s $session
tmux send-key -t $session:0 "WEBRTC_CONNECT=stun:$stunserver:3478 /app/peerconnection_client_headless --server $receiver --path /app" Enter
sleep 1000000