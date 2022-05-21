# importing the necessary dependencies
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
import os


import requests


# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "9Jtz2pQgrRaP9aY7Q73cl12vkzElGd8XLYHPKgGi6Mqn"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}




app=Flask(__name__)# initializing a flask app
#filepath="I:\SmartBridge Projects\Co2 emission\co2.pickle"
#model=pickle.load(open(co2.pickle,'rb'))

with open('co2.pkl', 'rb') as handle:
    model = pickle.load(handle)

@app.route('/')# route to display the home page
def home():
    return render_template('index.html') #rendering the home page
@app.route('/Prediction',methods=['POST','GET'])
def prediction(): # route which will take you to the prediction page
    return render_template('index1.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('index.html')

@app.route('/predict',methods=["POST","GET"])# route to show the predictions in a web UI
def predict():
    #  reading the inputs given by the user
    input_feature=[float(x) for x in request.form.values() ]  
    features_values=[np.array(input_feature)]
    feature_name=['CountryName', 'CountryCode', 'IndicatorName','Year']
    x=pd.DataFrame(features_values,columns=feature_name)
############################
payload_scoring = {"input_data":[{"fields":["CountryName","CountryCode","IndicatorName","Year"],"values":[input_feature]}]}


response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/b5199999-97b6-459a-810a-4b0d25fd138c/predictions?version=2022-03-07', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
print(response_scoring.json())


pred= response_scoring.json()
print(pred)
output = pred['predictions'][0]['values'][0][0]
print(output)
#########################
    
     # predictions using the loaded model file
    prediction=model.predict(x)  
    print("Prediction is:",prediction)
     # showing the prediction results in a UI
    return render_template("result.html",prediction=prediction[0])
if __name__=="__main__":
    
    # app.run(host='0.0.0.0', port=8000,debug=True)    # running the app
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=False,use_reloader=False)