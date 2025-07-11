import requests

url = "http://localhost:8000/detect/"
files = {"file": open("dataset_generation/images/images/img4.png", "rb")}
response = requests.post(url, files=files)
print(response.json())