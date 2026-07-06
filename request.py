import chardet
import requests


url = "http://127.0.0.1:5000/predict"
image_path = "test.png"

with open(image_path, "rb") as image_file:
    files = {"file": open("test.png", "rb")}
    response = requests.post(url, files=files)

print(f"Status Code: {response.status_code}")
print(f"Headers: {response.headers}")
print(f"Content: {response.text}")
print(response.json())