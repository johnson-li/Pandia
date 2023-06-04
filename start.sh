#!/bin/bash

duration=30
port=9999
rid=''
width=1080
fps=30

set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi
LONGOPTS=duration,port,id,width,fps
OPTIONS=d:p:i:w:f:
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
        -i|--id)
            rid="$2"
            shift 2
            ;;
        -f|--fps)
            fps="$2"
            shift 2
            ;;
        -w|--width)
            width="$2"
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

cd ~/Workspace/webrtc/src
# gn gen out/Default --args='is_debug=true rtc_use_h264=true ffmpeg_branding="Chrome" use_rtti=true'
ninja -C out/Default
cd -

rm /tmp/dump/* 2> /dev/null || true

session=pandia
tmux kill-session -t $session 2> /dev/null || true
tmux new-session -d -s $session
for i in `seq 1 5`
do
  tmux new-window -t ${session}:$i
done
tmux send-key -t $session:1 "~/Workspace/webrtc/src/out/Default/peerconnection_server --port ${port}" Enter
echo 'Server started'
sleep .1
tmux send-key -t $session:2 "~/Workspace/webrtc/src/out/Default/peerconnection_client_headless --port ${port} --name receiver --receiving_only true --force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/ 2> /tmp/pandia-receiver.log" Enter
echo 'Receiver started'
sleep 3
tmux send-key -t $session:3 "~/Workspace/webrtc/src/out/Default/peerconnection_client_headless --port ${port} --width ${width} --fps ${fps} --name sender --autocall true --force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/ 2> /tmp/pandia-sender.log" Enter
echo 'Sender started'
tmux send-key -t $session:4 'nload lo' Enter
tmux send-key -t $session:5 '' Enter

echo Wait $duration seconds...
sleep $duration
~/Workspace/Pandia/stop.sh
