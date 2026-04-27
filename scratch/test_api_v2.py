
import requests

url = "http://127.0.0.1:8000/chat/voice"
data = {"question": "Tôi đang rất lo lắng về tình hình cổ phiếu VIC."}
response = requests.post(url, data=data)

print(f"Status: {response.status_code}")
try:
    print(f"Response: {response.json()}")
except:
    print(f"Text: {response.text}")
