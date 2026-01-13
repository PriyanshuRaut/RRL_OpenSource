# Hardware Protocol Specification

## Overview
This document defines the serial protocol used by RRL hardware adapters (e.g., the Arduino adapter in `hardware.py`).
It includes message formats, expected responses, and versioning rules.

## Transport
- **Physical layer:** Serial (e.g., USB serial).
- **Encoding:** UTF-8 text.
- **Line endings:** Commands and responses are terminated by `\n` (LF). The adapter is tolerant of missing trailing newlines.

## Protocol Versioning
- Protocol versions are integer values: `proto=1`, `proto=2`, etc.
- **Backward compatibility:** Fields may be added to responses; unknown fields must be ignored.
- **Forward compatibility:** Clients must treat unknown features as unsupported.

## Message Formats
### Command Format
Commands are uppercase tokens with optional colon-separated arguments.

```
COMMAND[:ARG1[:ARG2...]]\n
```

Examples:
- `LED_ON:13\n`
- `MOTOR_START:9:255\n`

### Response Format
Structured responses use a key-value list prefixed by a response type.

```
TYPE:key=value;key=value;key=value\n
```

Example:
```
CAPS:proto=1;features=led_on,led_off,motor_start,motor_stop;device=arduino;model=uno\n
```

## Handshake & Capabilities
### CAPS Handshake
- **Command:** `CAPS\n`
- **Expected response:** `CAPS:proto=<int>;features=<csv>[;device=<str>][;model=<str>][;fw=<str>]\n`

Required fields:
- `proto`: protocol version integer.
- `features`: comma-separated feature list in lowercase.

Optional fields:
- `device`, `model`, `fw` (firmware version), and any future fields.

### Feature Names
Feature names are lowercase, snake_case tokens used for gating adapter methods:
- `led_on`
- `led_off`
- `motor_start`
- `motor_stop`

## Error Handling
- If a device does not support `CAPS`, it may return nothing or an unstructured message.
- Clients should treat missing or malformed responses as **unknown capabilities** and avoid assuming support for optional features.
