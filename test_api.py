import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    "ad_type": "Service-Based",
    "category": "Health & Fitness",
    "target_audience": "Teens",
    "platform": "Social Media",
    "Impressions": 30000,
    "Clicks": 5000,
    "ad_spend": 2000,
    "ad_duration": 10,
    "Location": "New York"
}

response = requests.post(url, json=data)
print(response.json())
