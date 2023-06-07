#!/bin/bash

while read -r line
do 
    IFS=' _'
    read -a array <<< "$line"
    port=${array[1]}
    echo $port
    ns=pandia_$port
    veth=veth$port
    vpeer=vpeer$port
    sudo ip netns del $ns 
    sudo ip link del $veth 
    sudo ip link del $vpeer 
done < <(sudo ip netns list)
