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
        Xây dựng môi trường cô lập, xóa các biến KAGGLE_* xung đột từ shell cha.
        """
        env = os.environ.copy()
        
        # Xóa các biến Kaggle toàn cục có thể gây xung đột
        keys_to_remove = [k for k in env if k.startswith("KAGGLE_")]
        for k in keys_to_remove:
            del env[k]
            
        # Đặt KAGGLE_CONFIG_DIR trỏ chuẩn xác tới thư mục profile
        env["KAGGLE_CONFIG_DIR"] = str(self.profile_dir.resolve())
        env["PYTHONUNBUFFERED"] = "1"
        
        # Nạp trực tiếp KAGGLE_USERNAME và KAGGLE_KEY vào env của subprocess
        # giúp tương thích 100% với cả Kaggle CLI lẫn Python Kaggle API client
        try:
            with open(self.kaggle_json, "r", encoding="utf-8") as f:
                kdata = json.load(f)
            if kdata.get("username"):
                env["KAGGLE_USERNAME"] = kdata["username"].strip()
            if kdata.get("key"):
                env["KAGGLE_KEY"] = kdata["key"].strip()
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
        Kiểm tra kết nối và tính hợp lệ của tài khoản.
        """
        try:
            res = self.run_command(["kernels", "list", "-m", "--page-size", "5"], timeout=30)
            if res.returncode == 0:
                return True, "Xác thực thành công với Kaggle API!"
            else:
                err_msg = res.stderr.strip() or res.stdout.strip()
                return False, f"Lỗi xác thực: {err_msg}"
        except subprocess.TimeoutExpired:
            return False, "Timeout khi kết nối tới Kaggle API (30s)!"
        except Exception as e:
            return False, str(e)


    def push_kernel(self, bundle_dir: Path) -> Dict[str, Any]:
        """
        Đẩy notebook từ bundle_dir lên Kaggle.
        """
        bundle_dir = Path(bundle_dir)
        meta_file = bundle_dir / "kernel-metadata.json"
        if not meta_file.exists():
            return {"success": False, "message": f"Thiếu kernel-metadata.json trong {bundle_dir}"}

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        kernel_id = meta.get("id", "Unknown")

        print(f"[*] Đang đẩy kernel '{kernel_id}' lên Kaggle qua profile cô lập...")
        res = self.run_command(["kernels", "push", "-p", str(bundle_dir.resolve())])

        output = res.stdout + "\n" + res.stderr
        if res.returncode == 0 and "Kernel version" in output:
            return {
                "success": True,
                "kernel_id": kernel_id,
                "url": f"https://www.kaggle.com/code/{kernel_id}",
                "output": output.strip()
            }
        else:
            # Kiểm tra xem có phải thành công nhưng thông báo khác không
            if "successfully pushed" in output.lower():
                return {
                    "success": True,
                    "kernel_id": kernel_id,
                    "url": f"https://www.kaggle.com/code/{kernel_id}",
                    "output": output.strip()
                }
            return {
                "success": False,
                "kernel_id": kernel_id,
                "error": output.strip(),
                "returncode": res.returncode
            }

    def get_status(self, kernel_id: str) -> str:
        """
        Lấy trạng thái thực thi hiện tại của kernel.
        """
        res = self.run_command(["kernels", "status", kernel_id])
        if res.returncode == 0:
            return res.stdout.strip()
        return res.stderr.strip() or res.stdout.strip()
