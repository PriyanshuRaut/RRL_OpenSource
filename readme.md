# RRL Syntax Guide (v2.0.0)

## Comments
Lines starting with `#` or containing `#` are comments.

```rrl
# This is a comment
x = 10 # Inline comment
````

---

## Variables

Variables can store numbers, strings, booleans, and expressions.

```rrl
x = 10
y = 2.5
name = "Raut"
flag = True
```

---

## Display (Output)

Use `display(...)` to print text or values.

```rrl
display("Hello, RRL!")
display("Sum:", x + y)
display(robot)
```

---

## Expressions

You can use math expressions and safe built-ins (`abs`, `min`, `max`, `round`, `int`, `float`, `str`, `bool`, `len`, `range`, `math.*`).
Imports are supported:

```rrl
import math
import os as myos
from math import sqrt, pi
```


```rrl
x = math.sqrt(16)
y = abs(-10)
z = max(1, 5, 3)
```

---

## Imports

RRL supports `import` and `from ... import ...` for a limited allowlist of modules. By default, only the following modules are allowed:

* `math`
* `hardware`
* `time`

Attempts to import any other module will raise a runtime error.

```rrl
import math
from math import sqrt
import hardware
```

---

## If / Elif / Else

Conditional blocks control execution flow.

```rrl
if x > 5
  display("x is big")
elif x == 5
  display("x is exactly 5")
else
  display("x is small")
endif
```

---

## Repeat Loop

Repeats the body exactly N times.

```rrl
repeat 3
  display("Hello!")
endrepeat
```

---

## While Loop

Repeats while the condition is true. Limited to **1,000,000 iterations** for safety.

```rrl
while x > 0
  display("Counting:", x)
  x = x - 1
endwhile
```

---

## Do While Loop

Runs the body at least once, then repeats while the condition is true. Limited to **1,000,000 iterations** for safety.

```rrl
do while x < 3
  display("x:", x)
  x = x + 1
endwhile
```

---

## For Loop

Iterates over any iterable expression (e.g., `range`, lists, tuples).

```rrl
for n in range(1, 4)
  display("n:", n)
endfor
```

---

## Functions

Define reusable code blocks with parameters and return values.

```rrl
def add(a, b)
  return a + b
enddef

result = add(10, 20)
display("Result:", result)
```

---

## Robot Simulator (`robot`)

A built-in object for simulating robot movement.

```rrl
robot.move(5)
robot.rotate(90)
robot.stop()
display(robot)
```

---

## Object-Oriented Programming (OOP)

RRL supports core OOP features including classes, methods, inheritance, and `self`.

```rrl
class Vehicle
  def __init__(name)
    self.name = name
  enddef

  def label()
    return "Vehicle: " + self.name
  enddef
endclass

class Robot(Vehicle)
  def __init__(name, speed)
    self.name = name
    self.speed = speed
  enddef

  def label()
    return "Robot: " + self.name + " @ " + str(self.speed)
  enddef
endclass

r = Robot("Atlas", 5)
display(r.label())
```

---

## Hardware Integration (Arduino & Raspberry Pi)

The `hardware` module exposes a friendly adapter for Arduino serial devices and Raspberry Pi GPIO. If the device is not available, methods return an error payload instead of raising.

### Arduino

```rrl
import hardware

result = hardware.led_on(13)
display(result)
hardware.motor_start(9, 200)
hardware.motor_stop(9)
```

Requires `pyserial` on the host system.

### Raspberry Pi

```rrl
import hardware

hardware.pi_setup_output(18)
hardware.pi_write(18, True)
hardware.pi_write(18, False)
hardware.pi_cleanup(18)
```

Requires `RPi.GPIO` on Raspberry Pi systems.

---

## REPL Commands

Special commands for the interactive REPL environment.

```
:help   → show help
:env    → show current variables & functions
:quit   → exit REPL
```

---

## Example Program

Complete example: move the robot in a square.

```rrl
def square(side)
  repeat 4
    robot.move(side)
    robot.rotate(90)
  endrepeat
enddef

square(5)
display("Done! Robot:", robot)
```

---

## Running RRL as an `.exe` Program

You can convert `rrl.py` into an executable and run `.rrl` files directly.

### Step 1: Install PyInstaller

```sh
pip install pyinstaller
```

### Step 2: Create the Executable

```sh
pyinstaller --onefile rrl.py
```

This will generate an `rrl.exe` file inside the `dist` folder.

### Step 3: Add to System PATH

* Go to **System Environment Variables → Path**
* Add the **dist folder path**

### Step 4: Run RRL Programs

* In **Command Prompt**:

  ```sh
  rrl.exe test.rrl
  ```

* In **VS Code** or any code editor:

  * Create a `.rrl` file
  * Run it with:

    ```sh
    rrl.exe filename.rrl
    ```

---

## License

RRL is released under the **MIT License**.

---


## Support & Sustainability

**RRL is developed and maintained as an open project.**

If you find it useful and want to support its continued development, documentation, and experimentation, you can do so via Patreon.

- Supporters may occasionally receive:
- Early previews of new features or syntax
- Development notes and design decisions
- Sneak peeks into upcoming tools or related projects

There are no fixed schedules or obligations. Support is primarily about enabling the work itself.

Patreon: https://patreon.com/PriyanshuRauth
