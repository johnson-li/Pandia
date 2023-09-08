FROM python:3.8

RUN apt update && apt install -y iproute2 gdb iputils-ping dnsutils net-tools tmux netcat-openbsd
COPY . /app/
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 9998-9999/tcp
EXPOSE 1-65535/udp

CMD python3 -u controller.py
