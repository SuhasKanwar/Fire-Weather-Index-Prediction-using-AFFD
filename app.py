import pickle
from  flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# importing the ridge regressor model and standard scaler pickle files
model = pickle.load(open('./models/regression.pkl', 'rb'))
scaler = pickle.load(open('./models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        temperature = float(request.form.get('temperature'))
        rh = float(request.form.get('rh'))
        ws = float(request.form.get('ws'))
        rain = float(request.form.get('rain'))
        ffmc = float(request.form.get('ffmc'))
        dmc = float(request.form.get('dmc'))
        isi = float(request.form.get('isi'))
        classes = float(request.form.get('classes'))
        region = float(request.form.get('region'))

        scaled_data = scaler.transform([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])
        prediction = model.predict(scaled_data)

        return render_template('predict.html', result=prediction[0])
    elif request.method == 'GET':
        return render_template('predict.html')
    else:
        return 'Invalid request method'

if __name__ == '__main__':
    app.run()