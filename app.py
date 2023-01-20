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



if __name__ == "__main__":
    app.run(debug=True)