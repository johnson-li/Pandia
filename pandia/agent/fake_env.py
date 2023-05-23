from multiprocessing import shared_memory
import time


def main():
    try:
        shm = shared_memory.SharedMemory(name='pandia', create=True, size=40)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name='pandia', create=False, size=40)

    offset = 0
    value = 1 * 1024
    shm.buf[offset * 4:offset * 4 + 4] = value.to_bytes(4, byteorder='little')

    value = 500 * 1024
    offset += 1
    shm.buf[offset * 4:offset * 4 + 4] = value.to_bytes(4, byteorder='little')

    value = 30
    offset += 1
    shm.buf[offset * 4:offset * 4 + 4] = value.to_bytes(4, byteorder='little')

    time.sleep(1000)

if __name__ == "__main__":
    main()