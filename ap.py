from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("rf_model.pkl")

# Initialize Flask app
app = Flask(__name__, template_folder="frontend")

@app.route("/")
def home():
    return render_template("index.html")  # Serve the frontend

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Get JSON data from frontend
        features = np.array(data["features"]).reshape(1, -1)  # Convert to NumPy array

        prediction = model.predict(features)[0]  # Get prediction

        return jsonify({"prediction": int(prediction)})  # Return result as JSON

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)