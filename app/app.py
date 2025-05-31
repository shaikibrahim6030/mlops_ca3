from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    age = int(request.form["age"])
    features = np.array([[area, bedrooms, age]])
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="Predicted House Price: ${:,.2f}".format(prediction[0]))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)