import multiprocessing
import sys
import tempfile
import threading
import time

import filelock
import pytest

from aesara.compile.compilelock import force_unlock, local_mem, lock_ctx


def test_compilelock_force_unlock():
    with tempfile.TemporaryDirectory() as dir_name:
        with lock_ctx(dir_name):
            dir_key = f"{dir_name}/lock"

            assert dir_key in local_mem._locks
            assert local_mem._locks[dir_key]

            force_unlock(dir_name)

            assert dir_key not in local_mem._locks

            # A sub-process forcing unlock...
            ctx = multiprocessing.get_context("spawn")
            p = ctx.Process(target=force_unlock, args=(dir_name,))
            p.start()
            p.join()

            assert dir_key not in local_mem._locks


def check_is_locked(dir_name, q):
    try:
        with lock_ctx(dir_name, timeout=0.1):
            q.put("unlocked")
    except filelock.Timeout:
        q.put("locked")


def get_subprocess_lock_state(ctx, dir_name):
    q = ctx.Queue()
    p = ctx.Process(target=check_is_locked, args=(dir_name, q))
    p.start()
    result = q.get()
    p.join()
    return result


def run_locking_test(ctx):

    with tempfile.TemporaryDirectory() as dir_name:
        assert get_subprocess_lock_state(ctx, dir_name) == "unlocked"

        # create a lock on the test directory
        with lock_ctx(dir_name):
            dir_key = f"{dir_name}/lock"
            assert dir_key in local_mem._locks
            assert local_mem._locks[dir_key]

            assert get_subprocess_lock_state(ctx, dir_name) == "locked"

            with lock_ctx(dir_name, timeout=0.1):
                assert get_subprocess_lock_state(ctx, dir_name) == "locked"

            assert get_subprocess_lock_state(ctx, dir_name) == "locked"

        assert get_subprocess_lock_state(ctx, dir_name) == "unlocked"


def test_locking_thread():
    import traceback

    with tempfile.TemporaryDirectory() as dir_name:

        def test_fn_1(arg):
            try:
                with lock_ctx(dir_name):
                    # Notify the outside that we've obtained the lock
                    arg.append(False)
                    while True not in arg:
                        time.sleep(0.5)
            except Exception:
                # Notify the outside that we done
                arg.append(False)
                # If something unexpected happened, we want to know what it was
                traceback.print_exc()

        def test_fn_2(arg):
            try:
                with lock_ctx(dir_name, timeout=0.1):
                    # If this can get the lock, then our file lock has failed
                    raise AssertionError()
            except filelock.Timeout:
                # It timed out, which means that the lock was still held by the
                # first thread
                arg.append(True)
            except Exception:
                # If something unexpected happened, we want to know what it was
                traceback.print_exc()

        res = []
        thread_1 = threading.Thread(target=test_fn_1, args=(res,))
        thread_2 = threading.Thread(target=test_fn_2, args=(res,))

        thread_1.start()

        # Make sure the first thread has obtained the lock
        while False not in res:
            time.sleep(0.5)

        thread_2.start()

        # The second thread should raise `filelock.Timeout`
        thread_2.join()
        assert True in res

        thread_1.join()
        assert not thread_1.is_alive()
        assert not thread_2.is_alive()


@pytest.mark.skipif(sys.platform != "linux", reason="Fork is only available on linux")
def test_locking_multiprocess_fork():
    ctx = multiprocessing.get_context("fork")
    run_locking_test(ctx)


def test_locking_multiprocess_spawn():
    ctx = multiprocessing.get_context("spawn")
    run_locking_test(ctx)
