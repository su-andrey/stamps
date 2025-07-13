import requests

url = "http://localhost:8000/detect/"
files = {"file": open("images/img4.png", "rb")}
response = requests.post(url, files=files, data={"conf_value": 0.25})
print(response.json())