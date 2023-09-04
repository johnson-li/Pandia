FROM continuumio/miniconda3

RUN apt update && apt install -y iproute2 gdb iputils-ping dnsutils net-tools tmux
COPY . /app/
WORKDIR /app

CMD ["/app/run.sh"]
