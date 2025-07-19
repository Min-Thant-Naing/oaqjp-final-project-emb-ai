from flask import Flask, request, jsonify
import joblib
import numpy as np
app = Flask(__name__)
# Load the pre-trained model
model = joblib.load('final_project/emotion_model.pkl')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'No features provided'}), 400
    
    features = np.array(data['features']).reshape(1, -1)
    
    # Predict the emotion
    prediction = model.predict(features)
    
    # Return the prediction as a JSON response
    return jsonify({'emotion': prediction[0]})
@app.route('/')
def index():
    return "Welcome to the Emotion Detection API! Use the /predict endpoint to get predictions."