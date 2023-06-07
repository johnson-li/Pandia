#!/bin/bash
set -o errexit -o pipefail -o noclobber -o nounset

period=10

delay_list=(10 20 50 80 100 200)
bw_list=(500 800 1000 2000 5000 8000 10000 20000 50000 80000 100000 200000 500000)

delay_size=${#delay_list[@]}
bw_size=${#bw_list[@]}


while true
do
    delay_index=$(($RANDOM % $delay_size))
    bw_index=$(($RANDOM % $bw_size))
    delay=${delay_list[$delay_index]}
    bw=${bw_list[$bw_index]}
    ~/Workspace/Pandia/scripts/start_traffic_control.sh -d $delay -b $bw
    echo "delay: $delay, bw: $bw"
    sleep $period
done
