import requests


# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "XFAZLsvV0OCZHhm0DtMIqj9GA_5M6F6CpZ6fIoKFW5ZZ"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line
input_feature=[float(x) for x in request.form.values() ]
payload_scoring = {"input_data":[{"fields":["CountryName","CountryCode","IndicatorName","Year"],"values":[input_feature]}]}
response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/4043e00c-5df0-4ad0-91ee-34d969c01ab0/predictions?version=2021-11-12', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
print(response_scoring.json())

pred= response_scoring.json()
print(pred)
output = pred['predictions'][0]['values'][0][0]
print(output)