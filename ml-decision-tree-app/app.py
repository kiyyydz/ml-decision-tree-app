from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")

# Daftar nama kolom sesuai training
fitur_form = ['Age', 'Gender', 'Occupation', 'Monthly Income', 'Feedback']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input dari form
        form_values = list(request.form.values())
        input_features = [float(x) for x in form_values]

        # Ubah ke DataFrame (agar cocok dengan pipeline)
        input_df = pd.DataFrame([input_features], columns=fitur_form)

        # Prediksi
        prediction = model.predict(input_df)
        output = "Yes" if prediction[0] == 1 else "No"

        return render_template("index.html", prediction_text=f"Prediksi Output: {output}")
    
    except Exception as e:
        return render_template("index.html", prediction_text=f"Terjadi error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
