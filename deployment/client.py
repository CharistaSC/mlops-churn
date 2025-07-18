# client.py
import requests

# Example dummy payload that mimics the dataset structure
payload = {
    "data": {
        "customerID": "4584-LBNMK",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 45,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 24.7,
        "TotalCharges": 1174.35
    }
}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
print("Raw Text:", response.text)