import pickle
import json

import numpy as np
import pandas as pd

from flask import Flask,request,jsonify,app,render_template,url_for
#create an application
app = Flask(__name__)
#let's make the data visible on browser
@app.route('/saketh')
def test():
    """Testing how data is returned"""
    return "Hey hai how are you"

#Load the model
regmodel = pickle.load(open("regmodel.pkl",'rb'))
scalar = pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])
    
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html",
                           prediction_text = "The House Price prediction is {}".format(output))



if __name__ == "__main__":
    app.run()
    
    
