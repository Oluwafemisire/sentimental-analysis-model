from flask import jsonify, Flask, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/', methods=['GET'])
def home():
    return jsonify({'service':'available'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    prediction = f"{prediction}"
    return jsonify({'sentiment': prediction})


if __name__ == '__main__':
    app.run()
