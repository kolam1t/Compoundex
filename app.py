"""
Compoundex - Country & Geopolitical Risk Predictor
Risk area: Country & Geopolitical Risk
Flask web application: HTML front end + JSON API endpoint.
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fintech.joblib")

FEATURES = [
    "sanctions_events_last_5_years",
    "sovereign_cds_spread_bps",
    "political_stability_index",
]

model = joblib.load(MODEL_PATH)


def predict(sanctions, cds, stability):
    """Return (prediction, probability) for one country-year."""
    x = pd.DataFrame(
        [[float(sanctions), float(cds), float(stability)]], columns=FEATURES
    )
    prediction = int(model.predict(x)[0])
    probability = float(model.predict_proba(x)[0][1])
    return prediction, probability


def risk_band(probability):
    """Map a probability to the country risk premium guidance shown to students."""
    if probability < 0.25:
        return "Low", "No additional country risk premium indicated."
    if probability < 0.50:
        return "Moderate", "Consider a modest country risk premium in the discount rate."
    if probability < 0.75:
        return "Elevated", "Apply a clear country risk premium and stress-test the valuation."
    return "High", "Apply a substantial country risk premium and review portfolio exposure."


# ----------------------------------------------------------------------
# HTML front end
# ----------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        try:
            sanctions = request.form["sanctions_events_last_5_years"]
            cds = request.form["sovereign_cds_spread_bps"]
            stability = request.form["political_stability_index"]
            prediction, probability = predict(sanctions, cds, stability)
            band, guidance = risk_band(probability)
            result = {
                "prediction": prediction,
                "label": "Elevated country risk" if prediction == 1 else "No elevated country risk",
                "probability": round(probability * 100, 1),
                "band": band,
                "guidance": guidance,
                "inputs": {
                    "sanctions_events_last_5_years": sanctions,
                    "sovereign_cds_spread_bps": cds,
                    "political_stability_index": stability,
                },
            }
        except (KeyError, ValueError):
            result = {"error": "Please enter valid numbers in all three fields."}
    return render_template("index.html", result=result)


# ----------------------------------------------------------------------
# API endpoint
# ----------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": "Missing required fields", "missing": missing}), 400
    try:
        prediction, probability = predict(
            data["sanctions_events_last_5_years"],
            data["sovereign_cds_spread_bps"],
            data["political_stability_index"],
        )
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric"}), 400

    band, guidance = risk_band(probability)
    return jsonify({
        "prediction": prediction,
        "label": "elevated_country_risk" if prediction == 1 else "no_elevated_country_risk",
        "probability": round(probability, 4),
        "risk_band": band,
        "guidance": guidance,
        "inputs": {f: data[f] for f in FEATURES},
        "model": "logistic_regression",
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None, "features": FEATURES})


if __name__ == "__main__":
    app.run(debug=True)
