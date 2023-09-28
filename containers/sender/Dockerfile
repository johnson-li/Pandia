FROM johnson163/pandia_sender_base

COPY . /app/
WORKDIR /app
EXPOSE 1-65535/udp

CMD python -m rllib_client
