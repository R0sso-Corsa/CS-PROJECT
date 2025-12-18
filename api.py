import requests
import json


url = "https://api.coingecko.com/api/v3/global"

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
else:
    print("Failed to retrieve data from the API. Status code:", response.status_code)

print(response.text)




#https://www.geeksforgeeks.org/saving-api-result-into-json-file-in-python/
