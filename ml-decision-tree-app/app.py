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
    try:
        # Ambil input dari form
        form_values = list(request.form.values())
        
        # Konversi ke float
        input_features = [float(x) for x in form_values]

        # Debug jumlah fitur (pastikan cocok dengan model kamu, misalnya 5 fitur)
        if len(input_features) != 5:
            raise ValueError(f"Jumlah fitur tidak cocok: {len(input_features)} fitur diberikan, harusnya 5.")

        # Buat array numpy
        final_features = np.array([input_features])

        # Prediksi
        prediction = model.predict(final_features)
        output = "Yes" if prediction[0] == 1 else "No"

        return render_template("index.html", prediction_text=f"Prediksi Output: {output}")
    
    except Exception as e:
        # Tampilkan pesan error ke user
        return render_template("index.html", prediction_text=f"Terjadi error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
