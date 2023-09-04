Pandia: Deep Reinforcement Learning for Real-time Video Streaming
===

Pandia uses a DRL model to improve the performance of video streaming. This repo implements the DRL model and the training/evaluation programs. The running of this repo requres a [customized WebRTC](https://github.com/johnson-li/webrtc/tree/pandia).

Train with RLLIB
===

Run on therminal 1:
```shell
python -m pandia.agent.env_server
```

Run on therminal 2:
```shell
python -m pandia.agent.env_client
```

Train with SB3
===

Not supported yet


Large-scale Training Infrastructure
===

