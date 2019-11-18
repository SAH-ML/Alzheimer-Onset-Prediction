import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

stack = pickle.load(open('stacker.pkl','rb'))

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [[float(x) for x in request.form.values()]]
    final_features = np.concatenate((int_features, stack.transform(int_features)), axis=1)
    #output = model.predict(final_features)
    output=model.decision_function(final_features)
    

    #output = prediction.reshape(-1, 1)

    return render_template('index.html', prediction_text='Chance of alzheimer disease {}%'.format(output))


if __name__ == "__main__":
    app.run(debug=True)