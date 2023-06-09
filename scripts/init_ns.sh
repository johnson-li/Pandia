#!/bin/bash
set -o errexit -o pipefail -o noclobber -o nounset

sudo sh -c 'echo 1 > /proc/sys/net/ipv4/ip_forward'
sudo iptables -P FORWARD DROP
sudo iptables -F FORWARD
sudo iptables -t nat -F

iface=eno1

function init_iptables () {
    id=$1
    port=70$id
    echo "init iptables for $port"
    ns=pandia$port
    veth=veth$port
    vpeer=vpeer$port
    veth_addr=10.200.$id.1
    vpeer_addr=10.200.$id.2

    sudo ip netns del $ns 2> /dev/null || true
    sudo ip link del $veth 2> /dev/null || true
    sudo ip link del $vpeer 2> /dev/null || true

    sudo ip netns add $ns
    sudo ip link add $veth type veth peer name $vpeer 2> /dev/null || true
    sudo ip link set $vpeer netns $ns
    sudo ip addr add $veth_addr/24 dev $veth
    sudo ip link set $veth up

    sudo ip netns exec $ns ip addr add $vpeer_addr/24 dev $vpeer
    sudo ip netns exec $ns ip link set $vpeer up
    sudo ip netns exec $ns ip link set lo up
    sudo ip netns exec $ns ip route add default via $veth_addr

    sudo iptables -t nat -A POSTROUTING -s $vpeer_addr/24 -o $iface -j MASQUERADE
    sudo iptables -A FORWARD -i $iface -o $veth -j ACCEPT
    sudo iptables -A FORWARD -o $iface -i $veth -j ACCEPT
    sudo iptables -t nat -A PREROUTING -i $iface -p tcp --dport $port -j DNAT --to-destination ${vpeer_addr}:$port
}

for i in `seq 1 9`
do
    if [[ 1 == ${#i} ]]; then
        i="0$i"
    fi
    init_iptables $i
done
