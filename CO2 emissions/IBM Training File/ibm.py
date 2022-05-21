import requests


# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "t_Q5MTvH2gPSslhfOrJg3d4Uxa5PlDcmjn5R9jaRcjda"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}


# NOTE: manually define and pass the array(s) of values to be scored in the next line
payload_scoring = {"input_data":[{"fields":["CountryName","CountryCode","IndicatorName","Year"],"values":[2005,12]}]}

response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/b5199999-97b6-459a-810a-4b0d25fd138c/predictions?version=2022-03-07', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
print(response_scoring.json())


pred= response_scoring.json()
print(pred)
output = pred['predictions'][0]['values'][0][0]
print(output)