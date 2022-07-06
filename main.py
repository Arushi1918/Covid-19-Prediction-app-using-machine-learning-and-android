
from flask import Flask, request, jsonify
import pickle
import numpy as np

model=pickle.load(open('model1.pkl','rb',))

app=Flask(__name__)
@app.route('/')
def home():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    cough = request.form.get('cough')
    fever = request.form.get('fever')
    sore_throat = request.form.get('sore_throat')
    shortness_of_breath = request.form.get('shortness_of_breath')
    headache = request.form.get('headache')
    age_60_and_above = request.form.get('age_60_and_above')
    gender = request.form.get('gender')
    test_indication = request.form.get('test_indication')

    input_query = np.array([[cough, fever, sore_throat, shortness_of_breath, headache,
              age_60_and_above, gender, test_indication]])

    result = model.predict(input_query)[0]

    return jsonify({'Covid Prediction Report': str(result)})

    return jsonify(result)

if __name__=='__main__':
    app.run(debug=True)

