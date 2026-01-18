from flask import request, jsonify
import numpy as np

@app.route('/predict', methods=['POST'])
def predict():
    """Predicts the Iris species based on input features"""
    try:
        data = request.get_json(force=True)
        features = [
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]
    except Exception as e:
        return jsonify({"error": str(e), "message": "Invalid input data. Please provide sepal_length, sepal_width, petal_length, and petal_width."}), 400

    # Convert features to a NumPy array for prediction
    features_array = np.array(features).reshape(1, -1)

    # Make prediction
    prediction_encoded = model.predict(features_array)
    prediction_label = le.inverse_transform(prediction_encoded)

    return jsonify({"prediction": prediction_label[0]})
