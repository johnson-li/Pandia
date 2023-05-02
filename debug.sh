cd ~/Workspace/webrtc/src
gn gen out/Default --args='is_debug=true rtc_use_h264=true ffmpeg_branding="Chrome" use_rtti=true'
ninja -C out/Default
cd -

sudo modprobe v4l2loopback devices=2
sudo chown lix16:lix16 /dev/video1
rm /tmp/dump/* 2> /dev/null

session=pandia
tmux kill-session -t $session 2> /dev/null
tmux new-session -d -s $session
for i in `seq 1 3`
do
  tmux new-window -t ${session}:$i
done
tmux send-key -t $session:0 'cd ~/Workspace/Pandia; python -m pandia.fakewebcam.main' Enter
tmux send-key -t $session:1 '~/Workspace/webrtc/src/out/Default/peerconnection_server' Enter
echo 'Server started'
sleep .1
tmux send-key -t $session:2 '~/Workspace/webrtc/src/out/Default/peerconnection_client_headless --name receiver --receiving_only true 2> ~/Workspace/Pandia/results/receiver.log' Enter
echo 'Receiver started'
sleep 3
tmux send-key -t $session:3 'gdb ~/Workspace/webrtc/src/out/Default/peerconnection_client_headless' Enter
echo 'Sender started'

