#!/bin/bash

bw=1048576 # bandwidth in kbps, 1Gbps by default
port=7018
delay=0 # ms
loss=0 # percentile
queue=250  # the queue size in ms
burst=1000  # in KB, should be at least bw / hz.
host=mobix

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

ssh $host "cd ~/Workspace/Pandia && ./scripts/start_traffic_control.sh -p $port -b $bw -d $delay -u $burst -q $queue -l $loss"
