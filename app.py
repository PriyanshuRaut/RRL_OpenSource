from flask import Flask, render_template, request, jsonify
import rrl  # assumes rrl.py is in the same folder

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run_code():
    data = request.json or {}
    code = data.get("code", "")
    out = []
    try:
        result = rrl.run_rrl_code(code, capture_output=out)
        env = result.get("env", {})
        # extract robot state safely for JSON
        robot = env.get("robot")
        robot_state = None
        if robot:
            robot_state = {
                "position": robot.position,
                "heading": robot.heading,
                "battery": robot.battery,
                "status": robot.status
            }
        return jsonify({"ok": True, "output": out, "robot": robot_state})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
