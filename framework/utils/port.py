import socket
import time


def check_used(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    if result == 0:
        sock.close()
        return True
    else:
        return False


def alloc(start_from: int = 7000) -> int:
    while True:
        if check_used(start_from):
            print("Port already used: %d" % start_from)
            start_from += 1
        else:
            return start_from


def wait_for(port: int, timeout: int = 5) -> bool:
    star_time = time.time()
    while not check_used(port):
        if time.time() - star_time > timeout:
            return False

        time.sleep(0.1)
    return True
