import requests
from requests.auth import HTTPBasicAuth

username = "bokhnhl"
key = "KGAT_26ef852621135f54947744308eee1793"

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
