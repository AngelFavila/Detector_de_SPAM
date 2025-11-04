from flask import Blueprint, render_template, request
import joblib
import os

main = Blueprint('main', __name__)

# Cargar modelo y vectorizador
model_path = os.path.join('app', 'model', 'spam_model.pkl')
vectorizer_path = os.path.join('app', 'model', 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    vectorized = vectorizer.transform([message])
    prediction = model.predict(vectorized)[0]
    result = "SPAM" if prediction == 1 else "No es spam"
    return render_template('index.html', prediction=result, message=message)
