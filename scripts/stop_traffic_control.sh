#!/bin/bash

sudo tc qdisc del dev veth1 handle ffff: ingress 2> /dev/null || true
sudo tc qdisc del dev veth1 root 2> /dev/null || true
sudo ip netns exec pandia tc qdisc del dev vpeer1 root 2> /dev/null || true