import urllib.request
import json

url = "http://127.0.0.1:8000/api/students/2de02e49-e727-44fc-8aca-e12fe732df3e"
data = {"gpa": 3.5, "attendance": 90.0, "program": "Engineering"}
req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), method='PUT')
req.add_header('Content-Type', 'application/json')

try:
    response = urllib.request.urlopen(req)
    print("Success:", response.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code)
    print("Body:", e.read().decode('utf-8'))
except Exception as e:
    print("Error:", e)
