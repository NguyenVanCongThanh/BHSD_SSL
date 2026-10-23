import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi

os.environ['KAGGLE_USERNAME'] = 'bokhnhl'
os.environ['KAGGLE_KEY'] = 'KGAT_26ef852621135f54947744308eee1793'

api = KaggleApi()
try:
    api.authenticate()
    print("[*] Authentication successful!")
except Exception as e:
    print("[X] Auth error:", e)

# Test pushing to see exact error message
try:
    res = api.kernels_push("/home/u001015/BHSD_SSL/kaggle_mean_teacher")
    print("[✓] Push result:", res)
except Exception as e:
    print("[X] Detailed Push Error:")
    print("Type:", type(e))
    print("Error:", e)
    if hasattr(e, 'response') and e.response is not None:
        print("Status code:", e.response.status_code)
        print("Response body:", e.response.text)
    if hasattr(e, 'error_message'):
        print("Error message:", e.error_message)
