
cd ~/Workspace/webrtc/src
gn gen out/Default --args='is_debug=false rtc_use_h264=true ffmpeg_branding="Chrome"'
ninja -C out/Default
cd -

session=pandia
tmux kill-session -t $session 2> /dev/null
tmux new-session -d -s $session
for i in `seq 1 5`
do
  tmux new-window -t ${session}:$i
done
tmux send-key -t $session:0 'sudo modprobe v4l2loopback devices=2; cd ~/Workspace/Pandia; python -m pandia.fakewebcam.main' Enter
tmux send-key -t $session:1 '~/Workspace/webrtc/src/out/Default/peerconnection_server' Enter
echo 'Server started'
sleep .1
tmux send-key -t $session:2 '~/Workspace/webrtc/src/out/Default/peerconnection_client_headless --name receiver --receiving_only true' Enter
echo 'Receiver started'
sleep 3
tmux send-key -t $session:3 '~/Workspace/webrtc/src/out/Default/peerconnection_client_headless --name sender --autocall true' Enter
echo 'Sender started'

