#!/bin/bash

host='mobix'

duration=30
port=7001
log='/dev/null'
dump=''

set -o errexit -o pipefail -o noclobber -o nounset
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

LONGOPTS=duration,port,log,dump
OPTIONS=d:p:l:u:
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
        -l|--log)
            log="$2"
            shift 2
            ;;
        -u|--dump)
            dump="$2"
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

ssh $host "cd ~/Workspace/Pandia && ./scripts/start_webrtc_receiver.sh -p $port -d $duration -l $log"
