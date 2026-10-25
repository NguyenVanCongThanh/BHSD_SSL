import os
import json
import requests

def check():
    with open("configs/kaggle/accounts.json") as f:
        accounts = json.load(f)["accounts"]
    
    for acc_id, acc in accounts.items():
        user = acc["username"]
        key = acc["key"]
        print(f"\n=== Checking account {acc_id}: {user} ===")
        headers = {
            "User-Agent": "Kaggle-MultiAccount-Manager/1.0",
            "Authorization": f"Bearer {key}"
        }
        
        # 1. Check kernels list
        r = requests.get(
            f"https://www.kaggle.com/api/v1/kernels/list?group=PROFILE&pagesize=5",
            headers=headers,
            timeout=15
        )
        if r.status_code == 200:
            kernels = r.json()
            for k in kernels:
                ref = k.get("ref")
                last_run = k.get("lastRunTime")
                print(f"Kernel: {ref} | Last run: {last_run}")
                
                # Check kernel status
                parts = ref.split("/")
                r_status = requests.get(
                    f"https://www.kaggle.com/api/v1/kernels/status?userName={parts[0]}&kernelSlug={parts[1]}",
                    headers=headers,
                    timeout=15
                )
                if r_status.status_code == 200:
                    status_data = r_status.json()
                    print(f"  -> Status: {status_data}")
        else:
            print(f"Error listing kernels: {r.status_code} {r.text}")

if __name__ == "__main__":
    check()
