Container Configuration
===


Emulator
===

![Architecture](https://docs.google.com/drawings/d/e/2PACX-1vRo5vBb13gPO2dweBHF4WKhud5dOoyGh3734hfA1iJNYhDVA9HTAjD7n5j45fXcdH3p1Vvt-gQmFwqn/pub?w=1175&h=432)

An example command of running the emulator container is as follow
```bash
docker run -d --rm --name sb3_emulator_d2d1a7e5 --hostname sb3_emulator_d2d1a7e5 --cap-add=NET_ADMIN --env NVIDIA_DRIVER_CAPABILITIES=all --runtime=nvidia --gpus all -v /tmp:/tmp --env PRIN
T_STEP=True -e SENDER_LOG=/tmp/sender.log --env NVENC=1 --env NVDEC=1 --env OBS_SOCKET_PATH=/tmp/d2d1a7e5_obs.sock --env LOGGING_PATH=None --env SB3_LOGGING_PATH=N
one --env CTRL_SOCKET_PATH=/tmp/d2d1a7e5_ctrl.sock johnson163/pandia_emulator python -um sb3_client
```

`cap-add=NET_ADMIN` allowes the container to create sockets visible to the host. It is essential for IPC.

`--env NVIDIA_DRIVER_CAPABILITIES=all --runtime=nvidia --gpus all` enables cuda support. It is essential when using hardware codec.

`-v /tmp:/tmp` shares the '/tmp' folder from the host to the container. It is essential for IPC.

`--env NVENC=1 --env NVDEC=1` configures whether to use NVENC or NVDEC.

`OBS_SOCKET_PATH` defines the path of the observation socket. WebRTC would write internal states to the socket and the python code would read from the socket.

`LOGGING_PATH and SB3_LOGGING_PATH` specifies the logging path of WebRTC. They could be set to a file inside '/tmp' so that the log is accessable from the host. It is mostly for debugging purposes.

`CTRL_SOCKET_PATH` specifies the socket file where the python code controls the WebRTC inside the container.

``