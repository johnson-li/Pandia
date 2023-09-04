#!/bin/bash

tc qdisc del dev eth0 root 2> /dev/null || true
tc qdisc add dev eth0 root netem delay 10ms
stunserver=$(dig +short stun)

echo Starting receiver, STUN server: $stunserver
session=pandia
tmux kill-session -t $session 2> /dev/null || true
tmux new-session -d -s $session
tmux send-key -t $session:0 "WEBRTC_CONNECT=stun:$stunserver:3478 /app/peerconnection_client_headless --receiving_only" Enter
sleep 1000000