#!/bin/bash

duration=30
port=7001

set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

LONGOPTS=duration,port,name
OPTIONS=d:p:
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

# Create network namespace before starting tmux session
iface=eno1
ns=pandia_$port
veth=veth$port
vpeer=vpeer$port
veth_addr=10.200.1.${port:(-2)}
vpeer_addr=10.200.1.1${port:(-2)}
if sudo ip netns list| grep $ns; then
    echo 'Network namespace already exists'
else
    sudo ip netns add $ns
    sudo ip link add $veth type veth peer name $vpeer 2> /dev/null || true
    sudo ip link set $vpeer netns $ns
    sudo ip addr add $veth_addr/24 dev $veth
    sudo ip link set $veth up

    sudo ip netns exec $ns ip addr add $vpeer_addr/24 dev $vpeer
    sudo ip netns exec $ns ip link set $vpeer up
    sudo ip netns exec $ns ip link set lo up
    sudo ip netns exec $ns ip route add default via $veth_addr

    sudo sh -c 'echo 1 > /proc/sys/net/ipv4/ip_forward'
    sudo iptables -P FORWARD DROP
    sudo iptables -F FORWARD
    sudo iptables -t nat -F
    sudo iptables -t nat -A POSTROUTING -s $vpeer_addr/24 -o $iface -j MASQUERADE
    sudo iptables -A FORWARD -i $iface -o $veth -j ACCEPT
    sudo iptables -A FORWARD -o $iface -i $veth -j ACCEPT
    sudo iptables -t nat -A PREROUTING -i ${iface} -p tcp --dport ${port} -j DNAT --to-destination ${vpeer_addr}:${port}
fi

session=pandia_$port
tmux kill-session -t $session 2> /dev/null || true
tmux new-session -d -s $session
for i in `seq 1 5`
do
  tmux new-window -t ${session}:$i
done
tmux send-key -t $session:1 "sudo ip netns exec pandia ~/Workspace/webrtc/src/out/Default/peerconnection_server --port ${port}" Enter
echo 'Server started'
sleep .1
tmux send-key -t $session:2 "sudo ip netns exec pandia ~/Workspace/webrtc/src/out/Default/peerconnection_client_headless --port ${port} --name receiver --receiving_only true --force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/ 2> /tmp/pandia-receiver.log" Enter

# Remove network namespace and kill tmux session after duration
tmux send-key -t $session:3 "sleep ${duration}; tmux kill-session -t ${session} 2> /dev/null" Enter
echo 'Receiver started'
