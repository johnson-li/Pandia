#!/bin/bash

bw=1048576 # bandwidth in kbps, 1Gbps by default
port=7001
delay=0 # ms
qlen=10000 # packets, 250ms of buffer, it is around 21 packets for 1 Mbps
loss=0 # percentile
host=mobix

set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

LONGOPTS=delay,bw,qlen,port,host,loss
OPTIONS=d:b:q:p:h:l:
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
        -l|--loss)
            loss="$2"
            shift 2
            ;;
        -h|--host)
            host="$2"
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

ssh $host "cd ~/Workspace/Pandia && ./scripts/start_traffic_control.sh -p $port -b $bw -d $delay -q $qlen -l $loss"
