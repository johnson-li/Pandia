FROM python:3.8

RUN apt update && apt install -y iproute2 gdb iputils-ping dnsutils net-tools tmux netcat-openbsd
COPY . /app/
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt

CMD python -m rllib_client
