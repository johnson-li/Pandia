Pandia: Deep Reinforcement Learning for Real-time Video Streaming
===

Pandia uses DRL to improve the video streaming performance of WebRTC. This repo implements the DRL model and the training/evaluation framework. The customized WebRTC is implemented in [another repo](https://github.com/johnson-li/webrtc/tree/pandia).

Architecture
===

![Architecture](https://docs.google.com/drawings/d/e/2PACX-1vTF7H__JOJsfnCUQKSdt9ubLGv_-BthUDodCtZBYpxiN45_XAmCTKZVTf3xfKW3BeBGxGDViAPCHezh/pub?w=957&h=375)


Compilation
===

The compilation is required once the container needs to be updated. Notice that the compilation script requires the WebRTC repo to be already setup locally.


```shell
./compile.sh
```

Running with RLlib
===
With RLlib, the containers run independently. The training is in a distributed manner. A cloud server is deployed for model training. Each streaming sender container has a local model for inference and is connected to the cloud server for updating the model, as well as the traning dataset.  

Start the training server:
```shell
python -m pandia.agent.env_server
```

Start the traning clients, i.e., webrtc containers:
```shell
docker compose up
# The number of training containers is specified when executing compile.sh 
```

Running with SB3
===

SB3 performs the traning in a centralized manner. All components run on the same machine. The training is provisioned by a single command. 

```shell
python -m pandia.train.train_sb3
```

To test the gym environment for sb3, run the following command:

```shell
python -m pandia.agent.env_container
```
