#!/bin/bash

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
