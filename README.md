Pandia: Open-source Framework for DRL-based Real-time Video Streaming Control
===
**NOTE: Please switch to the [mmsys24 branch](https://github.com/johnson-li/Pandia/tree/mmsys24) to check the submission code.**

Pandia uses DRL to improve the video streaming performance of WebRTC. This repo implements the DRL model and the training/evaluation framework. The customized WebRTC is implemented in [another repo](https://github.com/johnson-li/webrtc/tree/pandia). This repo invokes WebRTC via containers. So, there is no need to compile WebRTC by yourself for ordinary use.

Compilation (Optional)
===
Compilation is required only if modification has been made to WebRTC or its container. Refer to the [compilation guide](COMPILE.md)

Architecture
===

![Architecture](https://docs.google.com/drawings/d/e/2PACX-1vTF7H__JOJsfnCUQKSdt9ubLGv_-BthUDodCtZBYpxiN45_XAmCTKZVTf3xfKW3BeBGxGDViAPCHezh/pub?w=957&h=375)

Python scripts are used for training and evaluating the DRL model. The Python code uses IPC to read streaming reports from and sending streaming control settings to the container that runs WebRTC. 

Project Structure
===
```
Pandia
|--pandia   # The python codebase
|  |--agent  # Definition of the DRL agents
|  |  |--env_config.py  # Configuration of the agents 
|  |  |--env_emulator.py  # Definition of the emulator 
|  |--benchmark  # Python code of benchmarking the DRL agents
|  |--eval  # Python code of evaluting the DRL agents
|  |--train  # Python code of training the DRL agents
|  |--analysis  # Python code of analyzing the streaming performance
|  |  |--stream_illustrator.py  # Python code of generating performance analysis disgrams
|--containers  # The containers
|  |-- emulator  # The emulator container
|  |-- sender  # The WebRTC sender container
|  |-- receiver  # The WebRTC receiver container
|--hyperparams  # DRL training hyperparams used in SB3 zoo
|--scripts  # Python and bash scripts to automate the experiments
|--results  # The default place of the output diagrams
```


Environment Setup
===
The code is verified on Ubuntu 22.04.3 LTS and Python 3.8.17. It is recommended to use conda to manage the Python envs. Use the following command to install essential packets.

```bash
pip install -r requirements.txt
```



User Case 1: Emulator
===

Please check [EMULATOR.md](EMULATOR.md)


User Case 2: Dedicated Sender and Receiver
===

Please check [SENDER_RECEIVER.md](SENDER_RECEIVER.md)