from flask import Blueprint, request, jsonify
import joblib
import pandas as pd

api = Blueprint('api', __name__)

# Load the trained model
model = joblib.load('../models/model.pkl')

@api.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame(data, index=[0])
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction[0]})
