cd ~/Workspace/webrtc/src
gn gen out/Default --args='is_debug=true rtc_use_h264=true ffmpeg_branding="Chrome" use_rtti=true'
ninja -C out/Default
cd -

session=pandia
tmux kill-session -t $session 2> /dev/null
tmux new-session -d -s $session
for i in `seq 1 5`
do
  tmux new-window -t ${session}:$i
done
tmux send-key -t $session:0 'pulseaudio -D; sudo modprobe v4l2loopback devices=2; cd ~/Workspace/Pandia; python -m pandia.fakewebcam.main' Enter
tmux send-key -t $session:1 '~/Workspace/webrtc/src/out/Default/peerconnection_server' Enter
echo 'Server started'
sleep .1
tmux send-key -t $session:2 '~/Workspace/webrtc/src/out/Default/peerconnection_client_headless --name receiver --receiving_only true 2> ~/Workspace/Pandia/results/receiver.log' Enter
echo 'Receiver started'
sleep 3
tmux send-key -t $session:3 '~/Workspace/webrtc/src/out/Default/peerconnection_client_headless --name sender --autocall true 2> ~/Workspace/Pandia/results/sender.log' Enter
echo 'Sender started'
tmux send-key -t $session:4 'nload lo' Enter

echo 'Wait 30 seconds...'
sleep 30
~/Workspace/Pandia/stop.sh
