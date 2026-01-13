import logging
import serial  # for Arduino serial communication
import time


class HardwareError(Exception):
    """Base class for hardware-related errors."""


class HardwareConnectionError(HardwareError):
    """Raised when hardware connection is unavailable or fails."""


class HardwareWriteError(HardwareError):
    """Raised when a write to hardware fails."""


class HardwareReadError(HardwareError):
    """Raised when a read from hardware fails."""


class HardwareTimeoutError(HardwareError):
    """Raised when a hardware operation times out."""

class Arduino:
    def __init__(
        self,
        port="COM3",
        baudrate=9600,
        timeout=1,
        read_timeout=1.0,
        write_timeout=1.0,
        logger=None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.read_timeout = read_timeout
        self.write_timeout = write_timeout
        try:
            self.conn = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(2)  # wait for Arduino to reset
            self.logger.info(
                "[Arduino] Connected to %s at %s baud.", port, baudrate
            )
        except Exception as exc:
            self.logger.error("[Arduino] Connection failed: %s", exc)
            self.conn = None

    def write(self, message, timeout=None):
        """Send a message to Arduino."""
        if not self.conn:
            raise HardwareConnectionError("Arduino connection is not available.")

        effective_timeout = self.write_timeout if timeout is None else timeout
        previous_timeout = self.conn.write_timeout
        try:
            if effective_timeout is not None:
                self.conn.write_timeout = effective_timeout
            payload = str(message).encode()
            bytes_written = self.conn.write(payload)
            return {
                "ok": True,
                "bytes_written": bytes_written,
                "message": message,
            }
        except Exception as exc:
            raise HardwareWriteError(f"Failed to write to Arduino: {exc}") from exc
        finally:
            self.conn.write_timeout = previous_timeout

    def read(self, timeout=None, poll_interval=0.05):
        """Read a line from Arduino."""
        if not self.conn:
            raise HardwareConnectionError("Arduino connection is not available.")

        max_wait = self.read_timeout if timeout is None else timeout
        deadline = time.monotonic() + max_wait
        try:
            while time.monotonic() < deadline:
                if self.conn.in_waiting > 0:
                    data = self.conn.readline().decode().strip()
                    return {"ok": True, "data": data}
                time.sleep(poll_interval)
        except Exception as exc:
            raise HardwareReadError(f"Failed to read from Arduino: {exc}") from exc

        raise HardwareTimeoutError(
            f"Timed out waiting for Arduino data after {max_wait} seconds."
        )

    def led_on(self, pin=13):
        """Turn ON LED at given pin (default 13)."""
        return self.write(f"LED_ON:{pin}")

    def led_off(self, pin=13):
        """Turn OFF LED at given pin (default 13)."""
        return self.write(f"LED_OFF:{pin}")

    def motor_start(self, pin=9, speed=255):
        """Start motor at pin with speed (0-255)."""
        return self.write(f"MOTOR_START:{pin}:{speed}")

    def motor_stop(self, pin=9):
        """Stop motor at pin."""
        return self.write(f"MOTOR_STOP:{pin}")

    def close(self):
        """Close Arduino connection."""
        if self.conn:
            self.conn.close()
            self.logger.info("[Arduino] Connection closed.")
            return {"ok": True, "message": "Connection closed."}
        return {"ok": False, "message": "Connection already closed."}
