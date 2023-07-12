import time
import ntplib
from time import ctime

from pandia.ntp.ntpserver import system_to_ntp_time


def ntp_sync(server='195.148.127.230', port='8123'):
    c = ntplib.NTPClient()
    ts = time.time()
    response = c.request(server, port=port)
    return response


def main():
    c = ntplib.NTPClient()
    # response = c.request('pool.ntp.org')
    ts = time.time()
    # response = c.request('127.0.0.1', port='8123')
    response = c.request('195.148.127.230', port='8123')
    print(f'ts: {ts:.08f}, '
          f'precision: {response.precision:.08f}, offset: {response.offset:.08f}, '
          f'rtt: {response.delay:.08f}')


if __name__ == '__main__':
    main()
