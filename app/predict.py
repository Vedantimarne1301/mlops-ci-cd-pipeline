import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model at startup
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing features"}), 400

    features = np.array(data["features"]).reshape(1, -1)
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features).max())

    return jsonify({
        "prediction": prediction,
        "label": LABELS[prediction],
        "confidence": round(probability, 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)