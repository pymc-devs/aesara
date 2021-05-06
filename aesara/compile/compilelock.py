"""
Locking mechanism to ensure no two compilations occur simultaneously
in the same compilation directory (which can cause crashes).
"""
import os
import threading
from contextlib import contextmanager
from typing import Optional

import filelock

from aesara.configdefaults import config


__all__ = [
    "force_unlock",
    "lock_ctx",
]


class ThreadFileLocks(threading.local):
    def __init__(self):
        self._locks = {}
        # For each lock we also store the pid of the process
        # that created the lock, so we can create a new lock
        # after a fork
        self._pids = {}


local_mem = ThreadFileLocks()


def force_unlock(lock_dir: os.PathLike):
    """Forces the release of the lock on a specific directory.

    Parameters
    ----------
    lock_dir : os.PathLike
        Path to a directory that was locked with `lock_ctx`.
    """
    lockfile = os.path.join(lock_dir, "lock")
    fl = filelock.FileLock(lockfile)
    fl.release(force=True)

    if lockfile in local_mem._locks:
        del local_mem._locks[lockfile]


@contextmanager
def lock_ctx(lock_dir: os.PathLike = None, *, timeout: Optional[float] = None):
    """Context manager that wraps around FileLock and SoftFileLock from filelock package.

    Parameters
    ----------
    lock_dir : str
        A directory for which to acquire the lock.
        Defaults to the config.compiledir.
    timeout : float
        Timeout in seconds for waiting in lock acquisition.
        Defaults to config.compile__timeout.
    """
    if lock_dir is None:
        lock_dir = config.compiledir

    if timeout is None:
        timeout = config.compile__timeout

    # locks are kept in a dictionary to account for changing compiledirs
    lockfile = os.path.join(lock_dir, "lock")

    if lockfile in local_mem._locks and local_mem._pids[lockfile] == os.getpid():
        lock = local_mem._locks[lockfile]
    else:
        lock = filelock.FileLock(lockfile, timeout=timeout)
        local_mem._locks[lockfile] = lock
        local_mem._pids[lockfile] = os.getpid()

    with lock.acquire(timeout=timeout):
        yield
