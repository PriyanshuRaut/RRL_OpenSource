import importlib
import importlib.util
import time

try:
    import serial  # for Arduino serial communication
except Exception:  # pragma: no cover - optional dependency
    serial = None

class Arduino:
    def __init__(self, port="COM3", baudrate=9600, timeout=1, handshake=True, handshake_timeout=2.0):
        if serial is None:
            print("[Arduino] pyserial not installed; Arduino unavailable.")
            self.conn = None
        else:
            try:
                self.conn = serial.Serial(port, baudrate, timeout=timeout)
                time.sleep(2)  # wait for Arduino to reset
                print(f"[Arduino] Connected to {port} at {baudrate} baud.")
            except Exception as e:
                print("[Arduino] Connection failed:", e)
                self.conn = None
        self.capabilities = {"features": set(), "proto": None, "device": None, "model": None, "fw": None}
        self.capabilities_known = False
        if self.conn and handshake:
            self.handshake(timeout=handshake_timeout)

    def write(self, message):
        """Send a message to Arduino."""
        if self.conn:
            self.conn.write(str(message).encode())
            print(f"[Arduino] Sent: {message}")

    def read(self):
        """Read a line from Arduino if available."""
        return self._read_line(timeout=0)

    def _read_line(self, timeout=1.0):
        if not self.conn:
            return None
        end = time.time() + max(0.0, timeout)
        while True:
            if self.conn.in_waiting > 0:
                data = self.conn.readline().decode().strip()
                print(f"[Arduino] Received: {data}")
                return data
            if time.time() >= end:
                return None
            time.sleep(0.05)

    def _parse_caps_response(self, response):
        if not response or not response.startswith("CAPS:"):
            return None
        payload = response[len("CAPS:"):]
        fields = {}
        for part in payload.split(";"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
        if "proto" not in fields or "features" not in fields:
            return None
        try:
            proto = int(fields["proto"])
        except ValueError:
            return None
        features = {f.strip() for f in fields["features"].split(",") if f.strip()}
        return {
            "proto": proto,
            "features": features,
            "device": fields.get("device"),
            "model": fields.get("model"),
            "fw": fields.get("fw"),
        }

    def handshake(self, timeout=2.0):
        """Query device capabilities using the CAPS command."""
        if not self.conn:
            return None
        self.write("CAPS\n")
        response = self._read_line(timeout=timeout)
        parsed = self._parse_caps_response(response)
        if parsed:
            self.capabilities = parsed
            self.capabilities_known = True
            return parsed
        self.capabilities_known = False
        return None

    def get_capabilities(self):
        """Return last known capabilities, if any."""
        return dict(self.capabilities)

    def supports(self, feature):
        """Check if a feature is supported."""
        if not self.capabilities_known:
            return False
        return feature in self.capabilities["features"]

    def _require_feature(self, feature):
        if self.capabilities_known and not self.supports(feature):
            raise RuntimeError(f"[Arduino] Feature not supported: {feature}")

    def led_on(self, pin=13):
        """Turn ON LED at given pin (default 13)."""
        self._require_feature("led_on")
        self.write(f"LED_ON:{pin}")

    def led_off(self, pin=13):
        """Turn OFF LED at given pin (default 13)."""
        self._require_feature("led_off")
        self.write(f"LED_OFF:{pin}")

    def motor_start(self, pin=9, speed=255):
        """Start motor at pin with speed (0-255)."""
        self._require_feature("motor_start")
        self.write(f"MOTOR_START:{pin}:{speed}")

    def motor_stop(self, pin=9):
        """Stop motor at pin."""
        self._require_feature("motor_stop")
        self.write(f"MOTOR_STOP:{pin}")

    def close(self):
        """Close Arduino connection."""
        if self.conn:
            self.conn.close()
            print("[Arduino] Connection closed.")


class RaspberryPi:
    def __init__(self, mode="BCM"):
        self.mode = mode.upper()
        self.gpio = None
        self.available = False
        spec = importlib.util.find_spec("RPi.GPIO")
        if spec is not None:
            self.gpio = importlib.import_module("RPi.GPIO")
            if self.mode == "BOARD":
                self.gpio.setmode(self.gpio.BOARD)
            else:
                self.gpio.setmode(self.gpio.BCM)
            self.available = True

    def setup_output(self, pin):
        if not self.available:
            return False
        self.gpio.setup(pin, self.gpio.OUT)
        return True

    def setup_input(self, pin, pull="down"):
        if not self.available:
            return False
        pull = pull.lower()
        pud = self.gpio.PUD_DOWN if pull == "down" else self.gpio.PUD_UP
        self.gpio.setup(pin, self.gpio.IN, pull_up_down=pud)
        return True

    def write(self, pin, value):
        if not self.available:
            return False
        self.gpio.output(pin, self.gpio.HIGH if value else self.gpio.LOW)
        return True

    def read(self, pin):
        if not self.available:
            return None
        return bool(self.gpio.input(pin))

    def cleanup(self, pin=None):
        if not self.available:
            return False
        if pin is None:
            self.gpio.cleanup()
        else:
            self.gpio.cleanup(pin)
        return True


class HardwareAdapter:
    def __init__(self, port="COM3", baudrate=9600, timeout=1):
        self.arduino = Arduino(port=port, baudrate=baudrate, timeout=timeout)
        self.raspberry_pi = RaspberryPi()

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

    def _ensure_pi(self):
        if not getattr(self.raspberry_pi, "available", False):
            return {
                "ok": False,
                "error": {
                    "code": "pi_unavailable",
                    "message": "Raspberry Pi GPIO not available.",
                },
            }
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

    def pi_setup_output(self, pin):
        error = self._ensure_pi()
        if error:
            return error
        self.raspberry_pi.setup_output(pin)
        return self._ok()

    def pi_setup_input(self, pin, pull="down"):
        error = self._ensure_pi()
        if error:
            return error
        self.raspberry_pi.setup_input(pin, pull=pull)
        return self._ok()

    def pi_write(self, pin, value):
        error = self._ensure_pi()
        if error:
            return error
        self.raspberry_pi.write(pin, value)
        return self._ok()

    def pi_read(self, pin):
        error = self._ensure_pi()
        if error:
            return error
        value = self.raspberry_pi.read(pin)
        return self._ok(value=value)

    def pi_cleanup(self, pin=None):
        error = self._ensure_pi()
        if error:
            return error
        self.raspberry_pi.cleanup(pin=pin)
        return self._ok()
