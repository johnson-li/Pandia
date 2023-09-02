FROM nvidia/cuda:12.1.0-devel-ubi9

RUN yum install -y procps && rm -rf /var/cache
COPY . /app/
WORKDIR /app

CMD ["/app/peerconnection_client_headless", "--receiving_only"]
