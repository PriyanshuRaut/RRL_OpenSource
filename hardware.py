import serial  # for Arduino serial communication
import time

class Arduino:
    def __init__(self, port="COM3", baudrate=9600, timeout=1):
        try:
            self.conn = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(2)  # wait for Arduino to reset
            print(f"[Arduino] Connected to {port} at {baudrate} baud.")
        except Exception as e:
            print("[Arduino] Connection failed:", e)
            self.conn = None

    def write(self, message):
        """Send a message to Arduino."""
        if self.conn:
            self.conn.write(str(message).encode())
            print(f"[Arduino] Sent: {message}")

    def read(self):
        """Read a line from Arduino."""
        if self.conn and self.conn.in_waiting > 0:
            data = self.conn.readline().decode().strip()
            print(f"[Arduino] Received: {data}")
            return data
        return None

    def led_on(self, pin=13):
        """Turn ON LED at given pin (default 13)."""
        self.write(f"LED_ON:{pin}")

    def led_off(self, pin=13):
        """Turn OFF LED at given pin (default 13)."""
        self.write(f"LED_OFF:{pin}")

    def motor_start(self, pin=9, speed=255):
        """Start motor at pin with speed (0-255)."""
        self.write(f"MOTOR_START:{pin}:{speed}")

    def motor_stop(self, pin=9):
        """Stop motor at pin."""
        self.write(f"MOTOR_STOP:{pin}")

    def close(self):
        """Close Arduino connection."""
        if self.conn:
            self.conn.close()
            print("[Arduino] Connection closed.")


class HardwareAdapter:
    def __init__(self, port="COM3", baudrate=9600, timeout=1):
        self.arduino = Arduino(port=port, baudrate=baudrate, timeout=timeout)

    def _no_device_error(self):
        return {
            "ok": False,
            "error": {
                "code": "no_device",
                "message": "No hardware device connected.",
            },
        }

    def _ok(self, **payload):
        response = {"ok": True}
        response.update(payload)
        return response

    def _ensure_connected(self):
        if not getattr(self.arduino, "conn", None):
            return self._no_device_error()
        return None

    def write(self, message):
        error = self._ensure_connected()
        if error:
            return error
        self.arduino.write(message)
        return self._ok()

    def read(self):
        error = self._ensure_connected()
        if error:
            return error
        data = self.arduino.read()
        return self._ok(data=data)

    def led_on(self, pin=13):
        error = self._ensure_connected()
        if error:
            return error
        self.arduino.led_on(pin=pin)
        return self._ok()

    def led_off(self, pin=13):
        error = self._ensure_connected()
        if error:
            return error
        self.arduino.led_off(pin=pin)
        return self._ok()

    def motor_start(self, pin=9, speed=255):
        error = self._ensure_connected()
        if error:
            return error
        self.arduino.motor_start(pin=pin, speed=speed)
        return self._ok()

    def motor_stop(self, pin=9):
        error = self._ensure_connected()
        if error:
            return error
        self.arduino.motor_stop(pin=pin)
        return self._ok()

    def close(self):
        error = self._ensure_connected()
        if error:
            return error
        self.arduino.close()
        return self._ok()
