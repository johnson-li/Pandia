Emulator
===


The emulator is a gym environment named WebRTCEmulatorEnv that runs both WebRTC sender and receiver in a single container. The sender and the receiver communicate via the localhost interface and the network is shapped by tc. The implementation is in [pandia/agent/env_emulator.py](pandia/agent/env_emulator.py).


### Configur the Emulator
The configuration of the gym environment is presented in [pandia/agent/env_config.py](pandia/agent/env_config.py). Check the source code to find out the configuration options.

#### Actions
The enabled actions are defined in ENV_CONFIG['actions_keys']. 7 actions are currently supported. All actions not specified in ENV_CONFIG['actions_keys'] would use WebRTC's original algorithm for control.

#### Observations
The enabled actions are defined in ENV_CONFIG['observation_keys']. The definition of the observation keys are implemented in [pandia/monitor/monitor_block.py](pandia/monitor/monitor_block.py). Each observation key should have a property definition in the MonitorBlock class.

#### Rewards
The reward function is defined in [pandia/agent/reward.py](pandia/agent/reward.py).

#### Video Source
The video source is defined in ENV_CONFIG['video_source'], including the FPS and the resolution. Notice that this configuration controls the video input of WebRTC, which is different from the streaming control parameters in the actions.


### Test the Emulator
```bash
python -m pandia.agent.env_emulator
```
Check the source code to get more detail of the test setup. The testing code sets the bandwidth to 3 Mbps and changes the bitrate among 1 Mbps, 2 Mbps, and 3 Mbps. The performance metrics are printed every step and each step lasts for 100 ms.


### Train the Emulator with SB3
```bash
python -m pandia.train.train_sb3_emulator
```

By default, the trained model would be saved to ~/sb3_logs and the tensorboard log would be saved to ~/sb3_tensorboard. The training process can be monitored via tensorboard.


### Evaluate the Emulator with SB3
```bash
python -m pandia.eval.eval_sb3_emulator
```
Notice that the path of the trained model should be correctly set in the python code.


### Evaluate with GCC
```bash
python -m pandia.eval.eval_gcc
```