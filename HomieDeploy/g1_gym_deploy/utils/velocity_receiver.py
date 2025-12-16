"""Utility: UDP Velocity receiver running in a background thread.

The receiver listens on (HOST, PORT) for UDP packets containing a JSON
object with a ``velocity`` key. The most recent value is stored and can be
retrieved via ``get_velocity()``. The server is started in the constructor
and can be cleanly stopped using ``stop()`` or the context-manager protocol.

Example:
    recv = VelocityReceiver(host='0.0.0.0', port=10000)
    # incoming packets: b'{"velocity": [1.0, 0.0, 0.0]}'
    v = recv.get_velocity()
    recv.stop()

"""
from __future__ import annotations

import json
import time
import logging
import socket
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)


class VelocityReceiver:
    """Simple UDP server that listens for JSON messages and stores last velocity.

    Attributes:
        HOST (str): host to bind to (default: '0.0.0.0').
        PORT (int): UDP port to bind to (default: 7002).
    """

    HOST: str = "0.0.0.0"
    PORT: int = 7002

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, bufsize: int = 4096, timeout: float = 1.0) -> None:
        """Start the UDP receiver in a background thread.

        Args:
            host: Host to bind to. Uses class `HOST` if None.
            port: Port to bind to. Uses class `PORT` if None.
            bufsize: Maximum UDP packet size to receive.
            timeout: Socket timeout in seconds to allow graceful shutdown checks.
        """
        if host is not None:
            self.HOST = host
        if port is not None:
            self.PORT = port

        self._bufsize = bufsize
        self._timeout = timeout

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_velocity: Optional[Any] = None
        # time (seconds since epoch) when we last successfully received a velocity
        self._last_received_time: Optional[float] = None
        # inactivity timeout (seconds) after which last velocity is set to zeros
        self._inactivity_timeout = self._timeout

        self._start_server()

    def _start_server(self) -> None:
        """Create socket, bind and start the listener thread."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(self._timeout)
            sock.bind((self.HOST, self.PORT))
            self._sock = sock
        except Exception:
            logger.exception("Failed to bind UDP socket to %s:%s", self.HOST, self.PORT)
            raise

        self._thread = threading.Thread(target=self._serve_loop, daemon=True, name="VelocityReceiver")
        self._thread.start()
        logger.debug("VelocityReceiver started on %s:%s", self.HOST, self.PORT)

    def _serve_loop(self) -> None:
        """Thread target: receive UDP packets, parse JSON, store 'velocity' key."""
        assert self._sock is not None
        while not self._stop_event.is_set():
            try:
                data, addr = self._sock.recvfrom(self._bufsize)
            except socket.timeout:
                # on timeout, check inactivity and zero velocity if no update for inactivity timeout
                with self._lock:
                    if self._last_received_time is not None:
                        if time.time() - self._last_received_time > self._inactivity_timeout:
                            # set to zeros if not already zeros
                            if self._last_velocity != (0.0, 0.0, 0.0):
                                logger.debug("No velocity for %.3fs, zeroing last velocity", self._inactivity_timeout)
                                self._last_velocity = (0.0, 0.0, 0.0)
                continue
            except OSError:
                # Socket closed
                break

            try:
                # decode and parse JSON
                text = data.decode("utf-8")
                obj = json.loads(text)
                if "velocity" in obj:
                    logger.debug("Received velocity from %s: %s", addr, obj["velocity"])
                    with self._lock:
                        vel = obj["velocity"]
                        if isinstance(vel, list) and len(vel) == 3:
                            self._last_velocity = (float(vel[0]), float(vel[1]), float(vel[2]))
                            self._last_received_time = time.time()
                        else:
                            logger.warning("Invalid velocity format from %s: %s", addr, vel)
                else:
                    logger.debug("JSON received without 'velocity' key from %s: %s", addr, obj)
            except Exception:
                logger.exception("Failed to parse velocity message from %s: %r", addr, data)

    def get_velocity(self) -> Optional[tuple[float, float, float]]:
        """Return the last received velocity (thread-safe)."""
        with self._lock:
            return self._last_velocity

    def stop(self, wait: bool = True) -> None:
        """Stop the server and optionally wait for the thread to finish."""
        self._stop_event.set()
        # close the socket to unblock recv
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                logger.exception("Error closing socket")
            finally:
                self._sock = None

        if wait and self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.debug("VelocityReceiver stopped")

    def __enter__(self) -> "VelocityReceiver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - safe cleanup
        self.stop()


__all__ = ["VelocityReceiver"]


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.DEBUG)

    recv = VelocityReceiver(port=7002)
    try:
        while True:
            vel = recv.get_velocity()
            if vel is not None:
                print(f"Last velocity: vx={vel[0]:.2f}, vy={vel[1]:.2f}, vyaw={vel[2]:.2f}")
            else:
                print("No velocity received yet.")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping receiver...")
    finally:
        recv.stop()