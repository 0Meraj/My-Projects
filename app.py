from flask import Flask, render_template, request, jsonify
from detector import predict_url
import json, os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    submitted_url = ""
    if request.method == "POST":
        submitted_url = request.form["url"]
        result = predict_url(submitted_url)
    return render_template("index.html", result=result, submitted_url=submitted_url)

@app.route("/report")
def report():
    if not os.path.exists("report.json"):
        return jsonify({"error": "No report found. Please run train_model.py first."}), 404
    with open("report.json") as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    app.run(debug=True)