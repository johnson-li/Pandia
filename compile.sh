cd ~/Workspace/webrtc/src
# gn gen out/Default --args='is_debug=true rtc_use_h264=true ffmpeg_branding="Chrome" use_rtti=true'
gn gen out/Release --args='is_debug=false rtc_use_h264=true ffmpeg_branding="Chrome" use_rtti=true rtc_use_x11=false'
ninja -C out/Release -j$(nproc) peerconnection_client_headless

cp out/Release/peerconnection_client_headless ~/Workspace/Pandia/containers/receiver
cp out/Release/peerconnection_client_headless ~/Workspace/Pandia/containers/sender
cp out/Release/peerconnection_client_headless ~/Workspace/Pandia/bin

cd -

