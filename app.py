import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('linear_reg.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict_api', methods=["POST"])
def predict_api():
    data = request.json['data']
    print("Data -> ",data)
    print(np.array( list(data.values())).reshape(1,-1))
    new_transform_data = scaler.transform(np.array( list(data.values())).reshape(1,-1))
    output = model.predict(new_transform_data)
    print(output[0])

    return jsonify(output[0])

@app.route('/predict', methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(list(data)).reshape(1,-1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template("index.html",prediction_text= "The House Prediction is  {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)