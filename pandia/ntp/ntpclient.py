import ntplib
from time import ctime


def main():
    c = ntplib.NTPClient()
    # response = c.request('pool.ntp.org')
    response = c.request('127.0.0.1', port='8123')
    print(f'ts: {response.tx_time}, precision: {response.precision}')


if __name__ == '__main__':
    main()
