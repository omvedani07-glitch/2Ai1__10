from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# ✅ Load Model
MODEL_PATH = "model.pkl"

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    print("✅ Model Loaded Successfully")
except Exception as e:
    raise Exception(f"❌ Model loading failed: {e}")


# 🌍 Home Route
@app.route('/')
def home():
    return render_template('index.html')


# 🚀 Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 📥 Get Inputs
        pclass = int(request.form.get('Pclass', 0))
        sex = request.form.get('Sex', 'male')
        age = float(request.form.get('Age', 0))
        sibsp = int(request.form.get('SibSp', 0))
        parch = int(request.form.get('Parch', 0))
        fare = float(request.form.get('Fare', 0))

        # 🔁 Encode
        sex = 1 if sex == 'male' else 0

        # 🧠 Model Input
        features = np.array([[pclass, sex, age, sibsp, parch, fare]])

        # 🎯 Prediction
        prediction = model.predict(features)[0]

        # 📊 Probability (if supported)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(features)[0][1]
        else:
            prob = None

        return render_template(
            'result.html',
            prediction=prediction,
            probability=round(prob * 100, 2) if prob else None
        )

    except Exception as e:
        return render_template(
            'result.html',
            prediction="error",
            error=str(e)
        )


# ❤️ Health Check (important for Render)
@app.route('/health')
def health():
    return {"status": "ok"}


# 🚀 Run App
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
