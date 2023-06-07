#!/bin/bash
set -o errexit -o pipefail -o noclobber -o nounset

while read -r line
do
    IFS=' '
    read -a array <<< "$line"
    ns=${array[0]}
    port=${ns:6}
    veth=veth$port
    ns=pandia$port
    echo Delete ns, port: $port, veth: $veth, ns: $ns
    sudo ip link del $veth || true
    sudo ip netns del $ns || true
done < <(sudo ip netns list)
