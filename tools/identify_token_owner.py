#!/usr/bin/env python3
"""Identify which Kaggle account owns a provided API token without hardcoding secrets."""
import os
import sys
import requests
from requests.auth import HTTPBasicAuth

token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KAGGLE_KEY", "").strip()
if not token:
    print("Usage: python identify_token_owner.py <TOKEN>")
    print("   or: KAGGLE_KEY=<TOKEN> python identify_token_owner.py")
    sys.exit(1)

users_to_test = ["bokhnhl", "thnhnguynvncng"]
url = "https://www.kaggle.com/api/v1/datasets/list?mine=true"

for u in users_to_test:
    r = requests.get(url, auth=HTTPBasicAuth(u, token))
    print(f"Testing username '{u}' with token: HTTP {r.status_code}")
    if r.status_code == 200:
        print(f"--> TOKEN THỰC CHẤT THUỘC VỀ USER: {u}!")
        datasets = r.json()
        print(f"    Số datasets tìm thấy của {u}: {len(datasets)}")
        for d in datasets[:3]:
            print(f"    - {d.get('ref')}")
    else:
        print(f"    Response: {r.text[:100]}")
