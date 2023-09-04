#!/bin/bash
# exit when any command fails
set -e

cd ~/Workspace/webrtc/src
gn gen out/Default --args='is_debug=true rtc_use_h264=true ffmpeg_branding="Chrome" use_rtti=true rtc_use_x11=false'
gn gen out/Release --args='is_debug=false rtc_use_h264=true ffmpeg_branding="Chrome" use_rtti=true rtc_use_x11=false'
ninja -C out/Default -j$(nproc) peerconnection_client_headless
ninja -C out/Release -j$(nproc) peerconnection_client_headless

t=Release
# t=Default
cp out/${t}/peerconnection_client_headless ~/Workspace/Pandia/containers/receiver
cp out/${t}/peerconnection_client_headless ~/Workspace/Pandia/containers/sender


cd ~/Workspace/Pandia
docker compose build
