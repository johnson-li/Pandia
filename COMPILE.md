COMPILE
===

Follow this instructure to compile the WebRTC docker containers for Pandia.


Folder Structure
===
This project assumes the following folder structure:

```
$HOME
└───Workspace
    └───webrtc (the webrtc project)
       └───src
    └───Pandia (this project)
```

Some folder names are hardcoded in the compilation script. If you have a different folder structure, you will need to modify the script accordingly.

WebRTC Checkout
===
Please follow the instructions in the [WebRTC project](https://webrtc.github.io/webrtc-org/native-code/development/) to setup the development environment of WebRTC. Then, checkout our modification from this [repo](https://github.com/johnson-li/webrtc/tree/pandia).

Commands
===
Use the following command to compile WebRTC and build related docker containers.

```bash
./compile.sh
```

