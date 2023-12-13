import os
import fcntl
import time
import errno


class LockFile:
    def __init__(self, fname: str):
        self._fname = fname
        self._fd = None

    def acquire(self):
        self._fd=open(self._fname, "w")
        try:
            os.chmod(self._fname, 0o777)
        except PermissionError:
            # If another user created it already, we don't have the permission to change the access rights.
            # But it can be ignored because the creator already set it right.
            pass

        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except IOError as e:
                if e.errno != errno.EAGAIN:
                    raise
                else:
                    time.sleep(0.1)

    def release(self):
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        self._fd.close()
        self._fd = None

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
