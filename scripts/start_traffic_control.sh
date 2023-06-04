#!/bin/bash

delay=10 # ms
bw=1000 # kbps
qlen=210 # packets, 250ms of buffer, it is around 21 packets for 1000kbps

set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

LONGOPTS=delay,bw,qlen
OPTIONS=d:b:q:
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    exit 2
fi
eval set -- "$PARSED"
while true; do
    case "$1" in
        -d|--delay)
            delay="$2"
            shift 2
            ;;
        -b|--bw)
            bw="$2"
            shift 2
            ;;
        -q|--qlen)
            qlen="$2"
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

sudo tc qdisc del dev veth1 handle ffff: ingress 2> /dev/null || true
sudo tc qdisc del dev veth1 root 2> /dev/null || true
sudo ip netns exec pandia tc qdisc del dev vpeer1 root 2> /dev/null || true
# sudo tc qdisc add dev veth1 handle ffff: ingress
sudo tc qdisc add dev veth1 root netem delay ${delay}ms rate ${bw}kbit
sudo ip netns exec pandia tc qdisc add dev vpeer1 root netem delay ${delay}ms rate ${bw}kbit
# sudo tc qdisc add dev veth1 root tbf rate ${bw}kbit burst 1540 latency ${delay}ms # qlen ${qlen}
# sudo ip netns exec pandia tc qdisc add dev vpeer1 root tbf rate ${bw}kbit burst 1540 latency ${delay}ms

