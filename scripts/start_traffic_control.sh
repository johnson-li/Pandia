#!/bin/bash

bw=1048576 # bandwidth in kbps, 1Gbps by default
port=7001
delay=0 # ms
loss=0 # percentile
queue=250  # the queue size in ms
burst=1000  # in KB, should be at least bw / hz.

set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

LONGOPTS=delay,bw,port,loss,queue,burst
OPTIONS=d:b:p:l:q:u:
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
        -u|--burst)
            burst="$2"
            shift 2
            ;;
        -p|--port)
            port="$2"
            shift 2
            ;;
        -q|--queue)
            queue="$2"
            shift 2
            ;;
        -l|--loss)
            loss="$2"
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

echo Traffic control, bw: $bw kbps, delay: $delay ms, burst: $burst KB, loss: $loss '%', queue: $queue ms

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


# sudo tc qdisc add dev $veth root netem ${loss_clause} delay ${delay}ms rate ${bw}kbit
sudo tc qdisc add dev $veth root handle 1: netem delay ${delay}ms
sudo tc qdisc add dev $veth parent 1: handle 2: tbf rate ${bw}kbit burst ${burst}kb minburst 1540 latency ${queue}ms

exit

# sudo ip netns exec $ns tc qdisc add dev $vpeer root netem ${loss_clause} delay ${delay}ms rate ${bw}kbit
sudo ip netns exec $ns tc qdisc add dev $vpeer root handle 1: netem delay ${delay}ms
sudo ip netns exec $ns tc qdisc add dev $vpeer parent 1: handle 2: tbf rate ${bw}kbit burst ${burst}kb minburst 1540 latency ${queue}ms

