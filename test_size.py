
import requests

url = "http://127.0.0.1:8000/classify"

large_text = "A" * (2 * 1024 * 1024)

data = {
    "profile": "agent-harness-2026-09",
    "batchId": "test_size_limit",
    "policy": {},
    "cases": [
        {
            "caseId": "large_case",
            "messages": [
                {
                    "role": "user",
                    "content": large_text
                }
            ]
        }
    ]
}

response = requests.post(url, json=data)

print("HTTP Status:", response.status_code)
print("Response:", response.text)
