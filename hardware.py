import serial  # for Arduino serial communication
import time

class Arduino:
    def __init__(self, port="COM3", baudrate=9600, timeout=1, handshake=True, handshake_timeout=2.0):
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
