#!/bin/bash

bw=1048576 # bandwidth in kbps, 1Gbps by default
port=7001
delay=0 # ms
qlen=10000 # packets, 250ms of buffer, it is around 21 packets for 1 Mbps
loss=0 # percentile

set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

LONGOPTS=delay,bw,qlen,port
OPTIONS=d:b:q:p:
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

echo Traffic control, bw: $bw, delay: $delay, qlen: $qlen, loss: $loss

ns=pandia$port
veth=veth$port
vpeer=vpeer$port

sudo tc qdisc del dev $veth root 2> /dev/null || true
sudo ip netns exec $ns tc qdisc del dev $vpeer root 2> /dev/null || true

# sudo tc qdisc add dev $veth root handle 1: pfifo limit ${qlen}
if [[ $loss == 0 ]]
then
  loss_clause=''
else
  loss_clause="loss ${loss}%"
fi
sudo tc qdisc add dev $veth root netem ${loss_clause} delay ${delay}ms rate ${bw}kbit
sudo ip netns exec $ns tc qdisc add dev $vpeer root netem ${loss_clause} delay ${delay}ms rate ${bw}kbit

