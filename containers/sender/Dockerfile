FROM nvidia/cuda:12.1.0-devel-ubi9
COPY . /app/
WORKDIR /app

ENTRYPOINT ["peerconnection_client_headless", "--receiving_only"]
