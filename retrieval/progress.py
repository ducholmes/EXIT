"""Flushed progress and timed heartbeats for blocking preparation operations."""
import threading
import time
from contextlib import contextmanager


@contextmanager
def activity(label, details=None, interval=30):
    """A heartbeat reports elapsed time, not invented completion percentages."""
    started = time.monotonic()
    stopped = threading.Event()
    print(f'START: {label}', flush=True)

    def heartbeat():
        while not stopped.wait(interval):
            extra = f'; {details()}' if details else ''
            print(f'WORKING: {label}; elapsed {time.monotonic() - started:.0f}s{extra}', flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    except BaseException as exc:
        print(f'FAILED: {label} ({type(exc).__name__})', flush=True)
        raise
    else:
        print(f'DONE: {label}; elapsed {time.monotonic() - started:.1f}s', flush=True)
    finally:
        stopped.set()
        thread.join()


def copy_stream(source, target, label):
    copied = 0
    with activity(label, lambda: f'{copied / 1024**3:.2f} GiB written'):
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            target.write(block)
            copied += len(block)
