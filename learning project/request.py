import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "features": {
        "enginesize": 130,
        "curbweight": 2338,
        "horsepower": 111,
        "wheelbase": 98.8,
        "doornumber": 4,
        "cylindernumber": 4
    }
}

response = requests.post(url, json=payload)
print(response.json())