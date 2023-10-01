#!/bin/bash

set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
scale=1
LONGOPTS=scale
OPTIONS=s:
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    exit 2
fi
eval set -- "$PARSED"
while true; do
    case "$1" in
        -s|--scale)
            scale="$2"
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

for i in `seq 2 ${scale}` 
do
    if grep -Fxq "  sender$i:" compose.yaml
    then
        echo sender$i is already in compose.yaml
    else
        echo Add sender$i to compose.yaml
        echo "  receiver$i:" >> compose.yaml
        echo "    <<: *receiver1" >> compose.yaml
        echo "    hostname: receiver$i" >> compose.yaml
        echo "  sender$i:" >> compose.yaml
        echo "    <<: *sender1" >> compose.yaml
        echo "    hostname: sender$i" >> compose.yaml
        echo "    depends_on:" >> compose.yaml
        echo "      - stun" >> compose.yaml
        echo "      - receiver$i" >> compose.yaml
        echo "      - sender_base" >> compose.yaml

    fi
done

cd ~/Workspace/webrtc/src
gn gen out/Default --args='is_debug=true rtc_use_h264=true ffmpeg_branding="Chrome" use_rtti=true rtc_use_x11=false'
gn gen out/Release --args='is_debug=false rtc_use_h264=true ffmpeg_branding="Chrome" use_rtti=true rtc_use_x11=false'
ninja -C out/Default -j$(nproc) peerconnection_client_headless simulation
ninja -C out/Release -j$(nproc) peerconnection_client_headless simulation

for t in 'Release' 'Default'
do
    cp out/${t}/peerconnection_client_headless ~/Workspace/Pandia/containers/receiver/peerconnection_client_headless_${t}
    cp out/${t}/peerconnection_client_headless ~/Workspace/Pandia/containers/sender/peerconnection_client_headless_${t}
    cp out/${t}/simulation ~/Workspace/Pandia/containers/emulator/simulation_${t}
    cp -r ~/Workspace/Pandia/pandia ~/Workspace/Pandia/containers/sender
    cp -r ~/Workspace/Pandia/pandia ~/Workspace/Pandia/containers/emulator
done

cd ~/Workspace/Pandia
docker compose build
