# RRL Syntax Guide

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

```rrl
x = math.sqrt(16)
y = abs(-10)
z = max(1, 5, 3)
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