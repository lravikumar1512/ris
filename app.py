from flask import Flask, request, jsonify
import numpy as np
import joblib  # or pickle

app = Flask(__name__)

# Load trained model and label encoder
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")


@app.route('/predict', methods=['POST'])
def predict():
    """Predicts the Iris species based on input features"""
    data = request.get_json()

    if not data:
        return jsonify({
            "error": "No JSON data provided"
        }), 400

    try:
        features = np.array([
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]).reshape(1, -1)
    except KeyError as e:
        return jsonify({
            "error": f"Missing field: {e.args[0]}",
            "message": "Required fields: sepal_length, sepal_width, petal_length, petal_width"
        }), 400
    except ValueError:
        return jsonify({
            "error": "All feature values must be numeric"
        }), 400

    # Make prediction
    prediction_encoded = model.predict(features)
    prediction_label = le.inverse_transform(prediction_encoded)

    return jsonify({
        "prediction": prediction_label[0]
    })
