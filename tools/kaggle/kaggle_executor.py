#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kaggle_executor.py: Thực thi các lệnh Kaggle API / CLI trong môi trường cô lập
tuyệt đối (isolated sandbox environment). Loại bỏ hoàn toàn lỗi nhận nhầm tài khoản,
xung đột token toàn cục hay cache cấu hình của Python process.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

class KaggleExecutor:
    def __init__(self, profile_dir: Path):
        self.profile_dir = Path(profile_dir)
        self.kaggle_json = self.profile_dir / "kaggle.json"
        if not self.kaggle_json.exists():
            raise FileNotFoundError(f"Không tìm thấy file kaggle.json trong profile: {self.profile_dir}")
        self.env = self._build_isolated_env()
        self.kaggle_bin = self._resolve_kaggle_bin()

    def _resolve_kaggle_bin(self) -> str:
        # 1. Thử tìm trong PATH
        bin_path = shutil.which("kaggle")
        if bin_path:
            return bin_path
        
        # 2. Thử tìm trong ~/.local/bin/kaggle
        user_local = Path.home() / ".local" / "bin" / "kaggle"
        if user_local.exists() and os.access(user_local, os.X_OK):
            return str(user_local)
        
        # 3. Fallback vào lệnh kaggle mặc định
        return "kaggle"

    def _build_isolated_env(self) -> Dict[str, str]:
        """
        Xây dựng môi trường cô lập, xóa các biến KAGGLE_* xung đột từ shell cha
        và thiết lập chính xác các biến cần thiết cho tài khoản hiện tại.
        """
        env = os.environ.copy()
        
        # Xóa toàn bộ các biến Kaggle toàn cục có thể gây xung đột từ môi trường bên ngoài
        keys_to_remove = [k for k in env if k.startswith("KAGGLE_")]
        for k in keys_to_remove:
            del env[k]
            
        # Đặt KAGGLE_CONFIG_DIR trỏ chuẩn xác tới thư mục profile cô lập
        env["KAGGLE_CONFIG_DIR"] = str(self.profile_dir.resolve())
        env["PYTHONUNBUFFERED"] = "1"
        
        # Nạp trực tiếp KAGGLE_USERNAME, KAGGLE_KEY và KAGGLE_API_TOKEN vào env của subprocess
        # Hỗ trợ cả cơ chế Bearer Token mới (KGAT_...) và API Key truyền thống
        try:
            with open(self.kaggle_json, "r", encoding="utf-8") as f:
                kdata = json.load(f)
            if kdata.get("username"):
                env["KAGGLE_USERNAME"] = kdata["username"].strip()
            if kdata.get("key"):
                key_val = kdata["key"].strip()
                env["KAGGLE_KEY"] = key_val
                # Kaggle API Token (bắt đầu bằng KGAT_...) yêu cầu biến môi trường KAGGLE_API_TOKEN
                # để Kaggle CLI / API client tự động chuyển sang Bearer Authorization
                if key_val.startswith("KGAT_"):
                    env["KAGGLE_API_TOKEN"] = key_val
        except Exception:
            pass

        # Đảm bảo ~/.local/bin nằm trong PATH
        local_bin = str(Path.home() / ".local" / "bin")
        if local_bin not in env.get("PATH", ""):
            env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"
            
        return env

    def run_command(self, cmd_args: list, capture: bool = True, timeout: int = 40) -> subprocess.CompletedProcess:
        """
        Thực thi lệnh với môi trường cô lập, timeout và không chờ stdin.
        """
        full_cmd = [self.kaggle_bin] + cmd_args
        return subprocess.run(
            full_cmd,
            env=self.env,
            stdin=subprocess.DEVNULL,
            capture_output=capture,
            text=True,
            timeout=timeout
        )

    def test_connection(self) -> Tuple[bool, str]:
        """
        Kiểm tra kết nối và tính hợp lệ thực tế của tài khoản với Kaggle server.
        Khắc phục triệt để lỗi false-positive của api.authenticate().
        """
        with open(self.kaggle_json, "r", encoding="utf-8") as f:
            kdata = json.load(f)
        u = kdata.get("username", "").strip()
        k = kdata.get("key", "").strip()

        # 1. Kiểm tra trực tiếp qua HTTP REST call (chuẩn xác 100%)
        try:
            import requests
            headers = {"User-Agent": "Kaggle-MultiAccount-Manager/1.0"}
            auth = None
            if k.startswith("KGAT_"):
                headers["Authorization"] = f"Bearer {k}"
            else:
                auth = (u, k)
            
            # Gọi API lấy danh sách kernel profile để kiểm tra xác thực thực tế
            resp = requests.get(
                "https://www.kaggle.com/api/v1/kernels/list?group=PROFILE&pagesize=1",
                headers=headers,
                auth=auth,
                timeout=15
            )
            if resp.status_code == 200:
                return True, f"Xác thực thành công với tư cách {u}!"
            elif resp.status_code == 401:
                return False, f"Lỗi 401 Unauthorized: Token của '{u}' không hợp lệ hoặc đã bị thu hồi trên Kaggle."
            elif resp.status_code == 403:
                return False, f"Lỗi 403 Forbidden: Tài khoản '{u}' không có quyền truy cập."
            else:
                return False, f"HTTP Error {resp.status_code}: {resp.text[:120]}"
        except Exception as http_err:
            # 2. Fallback sang CLI subprocess nếu môi trường hạn chế thư viện requests
            try:
                res = self.run_command(["kernels", "list", "-m", "--page-size", "1"], timeout=20)
                if res.returncode == 0:
                    return True, f"Xác thực thành công với tư cách {u}!"
                err_msg = res.stderr.strip() or res.stdout.strip()
                return False, f"Lỗi xác thực: {err_msg}"
            except Exception as cli_err:
                return False, f"Lỗi kết nối: {http_err} | CLI: {cli_err}"

    def _push_via_rest(self, bundle_dir: Path, username: str, key: str, meta: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Đẩy notebook trực tiếp qua Kaggle REST API endpoint (POST /api/v1/kernels/push).
        Tương thích 100% với Bearer Token (KGAT_...) lẫn legacy Basic Auth.
        """
        import requests
        code_file_name = meta.get("code_file")
        code_path = bundle_dir / code_file_name
        if not code_path.exists():
            return False, f"Không tìm thấy file mã nguồn {code_file_name} trong bundle."

        with open(code_path, "r", encoding="utf-8") as f:
            code_text = f.read()

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Kaggle-MultiAccount-Manager/1.0"
        }
        auth = None
        if key.startswith("KGAT_"):
            headers["Authorization"] = f"Bearer {key}"
        else:
            auth = (username, key)

        payload = {
            "id": meta.get("id"),
            "slug": meta.get("id", "").split("/")[-1],
            "newTitle": meta.get("title"),
            "text": code_text,
            "language": meta.get("language", "python"),
            "kernelType": meta.get("kernel_type", "notebook"),
            "isPrivate": meta.get("is_private", True),
            "enableGpu": meta.get("enable_gpu", True),
            "enableTpu": meta.get("enable_tpu", False),
            "enableInternet": meta.get("enable_internet", True),
            "datasetDataSources": meta.get("dataset_sources", []),
            "kernelDataSources": meta.get("kernel_sources", []),
            "competitionDataSources": meta.get("competition_sources", []),
            "modelDataSources": meta.get("model_sources", [])
        }

        resp = requests.post(
            "https://www.kaggle.com/api/v1/kernels/push",
            headers=headers,
            auth=auth,
            json=payload,
            timeout=60
        )
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                data = {"message": resp.text}
            return True, data
        else:
            try:
                err_json = resp.json()
                err_msg = err_json.get("message", resp.text)
            except Exception:
                err_msg = resp.text
            return False, f"HTTP {resp.status_code}: {err_msg}"

    def push_kernel(self, bundle_dir: Path) -> Dict[str, Any]:
        """
        Đẩy notebook từ bundle_dir lên Kaggle bằng kiến trúc chịu lỗi 3 tầng:
        1. Kaggle REST API trực tiếp với Bearer Auth (Nhanh, chuẩn xác, không phụ thuộc cache)
        2. Kaggle CLI Subprocess trong môi trường cô lập đã nạp KAGGLE_API_TOKEN
        3. KaggleApi Python instance với Bearer headers được cấu hình rõ ràng
        """
        bundle_dir = Path(bundle_dir)
        meta_file = bundle_dir / "kernel-metadata.json"
        if not meta_file.exists():
            return {"success": False, "message": f"Thiếu kernel-metadata.json trong {bundle_dir}"}

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        kernel_id = meta.get("id", "Unknown")

        # Nạp thông tin xác thực từ profile cô lập
        with open(self.kaggle_json, "r", encoding="utf-8") as f:
            kdata = json.load(f)
        u = kdata.get("username", "").strip()
        k = kdata.get("key", "").strip()

        # TẦNG 1: Đẩy trực tiếp qua REST API (Hỗ trợ chuẩn Bearer Token KGAT_...)
        try:
            ok, rest_res = self._push_via_rest(bundle_dir, u, k, meta)
            if ok:
                url = rest_res.get("url") if isinstance(rest_res, dict) else None
                return {
                    "success": True,
                    "kernel_id": kernel_id,
                    "url": url or f"https://www.kaggle.com/code/{kernel_id}",
                    "output": str(rest_res)
                }
            rest_error = str(rest_res)
        except Exception as rest_exc:
            rest_error = str(rest_exc)

        # TẦNG 2: Fallback sang Kaggle CLI Subprocess (Môi trường đã nạp KAGGLE_API_TOKEN)
        try:
            res = self.run_command(["kernels", "push", "-p", str(bundle_dir.resolve())])
            output = (res.stdout + "\n" + res.stderr).strip()
            if res.returncode == 0 and ("Kernel version" in output or "successfully pushed" in output.lower()):
                return {
                    "success": True,
                    "kernel_id": kernel_id,
                    "url": f"https://www.kaggle.com/code/{kernel_id}",
                    "output": output
                }
            cli_error = output
        except Exception as cli_exc:
            cli_error = str(cli_exc)

        # TẦNG 3: Fallback sang KaggleApi Python với cấu hình Bearer Auth tường minh
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.set_config_value(api.CONFIG_NAME_USER, u, quiet=True)
            api.set_config_value(api.CONFIG_NAME_KEY, k, quiet=True)
            if k.startswith("KGAT_"):
                api.api_client.default_headers['Authorization'] = f"Bearer {k}"
            api.authenticate()
            if k.startswith("KGAT_"):
                api.api_client.default_headers['Authorization'] = f"Bearer {k}"
            push_res = api.kernels_push(str(bundle_dir.resolve()))
            return {
                "success": True,
                "kernel_id": kernel_id,
                "url": f"https://www.kaggle.com/code/{kernel_id}",
                "output": str(push_res)
            }
        except Exception as api_exc:
            api_error = str(api_exc)

        return {
            "success": False,
            "kernel_id": kernel_id,
            "error": f"Lỗi push qua cả 3 phương thức:\n  [1] REST: {rest_error}\n  [2] CLI: {cli_error}\n  [3] API: {api_error}"
        }

    def get_status(self, kernel_id: str) -> str:
        """
        Lấy trạng thái thực thi hiện tại của kernel.
        """
        # 1. Thử qua CLI
        res = self.run_command(["kernels", "status", kernel_id])
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
            
        # 2. Fallback sang REST API
        try:
            import requests
            with open(self.kaggle_json, "r", encoding="utf-8") as f:
                kdata = json.load(f)
            u = kdata.get("username", "").strip()
            k = kdata.get("key", "").strip()
            headers = {"User-Agent": "Kaggle-MultiAccount-Manager/1.0"}
            auth = None
            if k.startswith("KGAT_"):
                headers["Authorization"] = f"Bearer {k}"
            else:
                auth = (u, k)

            parts = kernel_id.split("/")
            user_part = parts[0] if len(parts) > 1 else u
            slug_part = parts[1] if len(parts) > 1 else parts[0]
            resp = requests.get(
                f"https://www.kaggle.com/api/v1/kernels/status?userName={user_part}&kernelSlug={slug_part}",
                headers=headers,
                auth=auth,
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("status", str(data))
        except Exception:
            pass

        return res.stderr.strip() or res.stdout.strip()

    def get_log(self, kernel_id: str) -> Tuple[bool, str]:
        """
        Lấy nội dung raw log của session hiện tại từ Kaggle REST API mà không cần tải
        checkpoint hay output files lớn (vài GB).
        """
        try:
            import requests
            with open(self.kaggle_json, "r", encoding="utf-8") as f:
                kdata = json.load(f)
            u = kdata.get("username", "").strip()
            k = kdata.get("key", "").strip()
            headers = {"User-Agent": "Kaggle-MultiAccount-Manager/1.0"}
            auth = None
            if k.startswith("KGAT_"):
                headers["Authorization"] = f"Bearer {k}"
            else:
                auth = (u, k)

            parts = kernel_id.split("/")
            user_part = parts[0] if len(parts) > 1 else u
            slug_part = parts[1] if len(parts) > 1 else parts[0]
            resp = requests.get(
                f"https://www.kaggle.com/api/v1/kernels/output?userName={user_part}&kernelSlug={slug_part}",
                headers=headers,
                auth=auth,
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                raw_log = data.get("log") or ""
                if not raw_log:
                    return False, "Không có nội dung log nào được ghi nhận cho kernel này."
                try:
                    events = json.loads(raw_log)
                    if isinstance(events, list):
                        plain_text = "".join(e.get("data", "") for e in events if isinstance(e, dict) and "data" in e)
                        return True, plain_text
                except Exception:
                    pass
                return True, raw_log
            elif resp.status_code == 404:
                return False, f"Lỗi 404: Không tìm thấy kernel '{kernel_id}' hoặc chưa có output session."
            else:
                return False, f"Lỗi HTTP {resp.status_code} khi tải log: {resp.text[:200]}"
        except Exception as exc:
            return False, f"Lỗi kết nối khi tải log: {exc}"

