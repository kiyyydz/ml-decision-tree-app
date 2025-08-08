from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    final_features = np.array([input_features])
    prediction = model.predict(final_features)
    
    output = "Yes" if prediction[0] == 1 else "No"
    return render_template("index.html", prediction_text=f"Prediksi Output: {output}")

if __name__ == '__main__':
    app.run(debug=True)
