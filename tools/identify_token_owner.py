import requests
from requests.auth import HTTPBasicAuth

token = "KGAT_26ef852621135f54947744308eee1793"
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
