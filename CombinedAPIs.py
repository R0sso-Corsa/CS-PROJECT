import requests
import json


url = "https://api.coingecko.com/api/v3/exchange_ratess/"

headers = {
    "accept": "application/json",
    "x-cg-demo-api-key": "CG-k6ayxiKmDjSQTgewkTDTA2Eh "
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    new_data = response.json()

    try:
        with open("data.json", "r") as json_file:
            existing_data = json.load(json_file)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        existing_data = []
    
    existing_data.append(new_data)

    with open("data.json", "w") as json_file:
        json.dump(existing_data, json_file, indent=4)
        print("Data appended to data.json file.")
        print(response.text)
else:
    print("Failed to retrieve data from the API. Status code:", response.status_code)
    API_KEY = "AIzaSyD3ucqysaQsJY7MIdG1cJ8ygF6tP6LXRzM"
    SEARCH_QUERY = "wadwadawdawdwdaddawdawdsadawd123123123213wadd"
    CSE_ID = "704686a19399c49b5"
    url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&q={SEARCH_QUERY}"
    if CSE_ID:
      url += f"&cx={CSE_ID}"
    try:
      response = requests.get(url)
      response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
      data = response.json()
      print(json.dumps(data, indent=4))  # Print the results in a readable format

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        
print(response.text)


#1 create json file
#2 read json file for total results
#3 if totalresults: 0, return error flag, else: print json.




#https://www.geeksforgeeks.org/saving-api-result-into-json-file-in-python/



# GITHUB PERSONAL API TOKEN: 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