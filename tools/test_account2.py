#!/usr/bin/env python3
"""Test account 2 Kaggle API connection without hardcoding credentials."""
import os
import sys
from pathlib import Path
import requests
from requests.auth import HTTPBasicAuth

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.kaggle.accounts_manager import AccountsManager

mgr = AccountsManager()
acc = mgr.get_account("2")
if not acc:
    print("[!] Account 2 not found in accounts config.")
    sys.exit(1)

username, key = mgr.resolve_credentials(acc)
if not key:
    key = os.environ.get("KAGGLE_KEY", "").strip()

if not username or not key:
    print(f"[!] Missing credentials for account 2 ({username}). Set KAGGLE_KEY or configure ~/.kaggle/kaggle.json")
    sys.exit(1)

url = "https://www.kaggle.com/api/v1/kernels/list?mine=true"
resp = requests.get(url, auth=HTTPBasicAuth(username, key))

print("Status code:", resp.status_code)
if resp.status_code == 200:
    kernels = resp.json()
    print(f"Authenticated successfully as {username}!")
    print(f"Number of kernels found on this account: {len(kernels)}")
    for k in kernels:
        print(" - Ref:", k.get('ref'), "| Title:", k.get('title'))
else:
    print("Response text:", resp.text)
