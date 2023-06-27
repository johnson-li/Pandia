#!/bin/bash

duration=10
port=7018
width=1080
fps=30

set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi
LONGOPTS=duration,port,width,fps
OPTIONS=d:p:w:f:
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

# Start receiver
~/Workspace/Pandia/scripts/start_webrtc_receiver_remote.sh -p $port -d $duration -l /tmp/test_receiver.log

# Init traffic control
~/Workspace/Pandia/scripts/start_traffic_control_remote.sh -p $port -b 1000000 -d 0

# Start sender
echo "Runnig... Will last for $duration s"
rm /tmp/test_sender.log 2> /dev/null
~/Workspace/Pandia/bin/peerconnection_client_headless --server 195.148.127.230 --port $port --width $width --fps $fps --name sender --autocall true --force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/ 2> /tmp/test_sender.log

# Copy receiver log
scp mobix:/tmp/test_receiver.log /tmp
