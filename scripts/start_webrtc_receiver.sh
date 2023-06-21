#!/bin/bash

duration=300
port=7001
log=/dev/null

set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

LONGOPTS=duration,port,name,log
OPTIONS=d:p:l:
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    exit 2
fi
eval set -- "$PARSED"
while true; do
    case "$1" in
        -d|--duration)
            duration="$2"
            shift 2
            ;;
        -p|--port)
            port="$2"
            shift 2
            ;;
        -l|--log)
            log="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

ns=pandia$port
uuid=`uuidgen`
echo "UUID: $uuid, port: $port, duration: $duration"

session=pandia$port
tmux kill-session -t $session 2> /dev/null || true
tmux new-session -d -s $session
for i in `seq 1 5`
do
  tmux new-window -t ${session}:$i
done
# tmux send-key -t $session:0 "cd ~/Workspace/Pandia; python pandia.ntp.ntpserver" Enter
tmux send-key -t $session:1 "sudo ip netns exec ${ns} ~/Workspace/webrtc/src/out/Default/peerconnection_server --port ${port}" Enter
echo 'Server started'
tmux send-key -t $session:2 "sudo ip netns exec ${ns} ~/Workspace/webrtc/src/out/Default/peerconnection_client_headless --port ${port} --name receiver --receiving_only true --force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/ 2> $log" Enter

# Remove network namespace and kill tmux session after duration
tmux send-key -t $session:3 "sleep ${duration}; tmux kill-session -t ${session} 2> /dev/null" Enter
echo 'Receiver started'
