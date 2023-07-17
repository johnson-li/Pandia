import time
import ntplib
from time import ctime

from pandia.ntp.ntpserver import system_to_ntp_time

NTP_OFFSET_PATH = '/tmp/ntp_offset.log'

def ntp_sync(server='195.148.127.230', port='8123'):
    c = ntplib.NTPClient()
    response = c.request(server, port=port)
    return response


def main():
    ts = time.time()
    ntp_resp = ntp_sync()
    print(f'NTP response: precision: {ntp_resp.precision}'
            f', offset: {ntp_resp.offset}, rtt: {ntp_resp.delay}')
    with open(NTP_OFFSET_PATH, 'w+') as f:
        f.write(f'{ntp_resp.offset}, {ntp_resp.delay}')


if __name__ == '__main__':
    main()
