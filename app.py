from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

# Inisialisasi Flask app
app = Flask(__name__)

# Load model .h5
model = load_model("Skinalyze.h5")

@app.route("/")
def home():
    return "Welcome to the Skinalyze API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Mendapatkan input dari request JSON
        data = request.json
        input_data = np.array(data["input"]).reshape(1, -1)  # Sesuaikan format input model
        prediction = model.predict(input_data).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)  # Port 8080 digunakan oleh Vercel
