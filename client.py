import requests

url = "http://localhost:8000/detect/"
files = {"file": open("images/img5.png", "rb")}
response = requests.post(url, files=files, data={"conf_value": 0.9}).json()
print(f"Найдено {len(response['detections'])} объекта")
print(response['detections'])
