"""POSIX process isolation for synchronous dependencies.

Kills the local worker process group; cannot cancel work already accepted by a
remote database/provider. Cleanup/finalization adds scheduler-dependent latency,
not a real-time scheduling guarantee. Fork isolation requires a single-threaded
caller; unsupported hosts fail closed rather than silently using threads.
"""
from __future__ import annotations

import os
import pickle
import signal
import socket
import struct
import threading

MAX_FRAME = 16 * 1024 * 1024


def send_frame(sock, value):
    data = pickle.dumps(value, protocol=4)
    if len(data) > MAX_FRAME:
        raise ValueError("worker snapshot exceeds IPC bound")
    sock.sendall(struct.pack("!I", len(data)) + data)


def receive_frame(sock, remaining=None):
    def exact(size):
        chunks = bytearray()
        while len(chunks) < size:
            if remaining is not None:
                timeout = remaining()
                if timeout <= 0:
                    raise socket.timeout("absolute worker deadline")
                sock.settimeout(timeout)
            part = sock.recv(size - len(chunks))
            if not part:
                raise EOFError("worker channel closed")
            chunks.extend(part)
        return chunks
    size = struct.unpack("!I", exact(4))[0]
    if size > MAX_FRAME:
        raise ValueError("worker frame exceeds IPC bound")
    return pickle.loads(exact(size))


class WorkerMeter:
    """Serialized RPC: the parent owns ALL admissions/settlements, not a mirror.

    A killed worker can never lose an admitted lease or overwrite newer counters.
    One lock spans each request/reply, including calls from P3 pool threads.
    """
    def __init__(self, sock, budget):
        self._socket, self.budget = sock, budget
        self._lock = threading.RLock()

    def _call(self, name, *args, **kwargs):
        with self._lock:
            send_frame(self._socket, ("meter", name, args, kwargs))
            ok, result = receive_frame(self._socket)
            if not ok:
                raise result
            return result

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: self._call(name, *args, **kwargs)

    def publish(self, kind, state):
        with self._lock:
            send_frame(self._socket, (kind, state))
            ok, result = receive_frame(self._socket)
            if not ok:
                raise result


def require_isolation_support():
    if os.name != "posix" or not hasattr(os, "fork"):
        raise RuntimeError("hard worker deadline requires POSIX process groups")
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise RuntimeError("fork-isolated runner requires a single-threaded main-thread caller")
    # Python's thread registry misses BLAS/extension/native threads. Never fork
    # an apparently single-threaded interpreter with native workers alive.
    import sys
    if sys.platform.startswith("linux"):
        native_threads = len(os.listdir("/proc/self/task"))
    elif sys.platform == "darwin":
        import subprocess
        listing = subprocess.run(["/bin/ps", "-M", "-p", str(os.getpid())],
                                 capture_output=True, text=True, check=True).stdout
        native_threads = max(0, len(listing.strip().splitlines()) - 1)
    else:
        raise RuntimeError("cannot verify native thread safety on this POSIX host")
    if native_threads != 1:
        raise RuntimeError("fork-isolated runner requires a single-threaded native process")


def kill_group(process):
    """SIGKILL (not a cooperative terminate), then reap the owned process."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    # Before setsid completed, the child is not yet a group leader.
    if process.is_alive():
        process.kill()
    process.join()
