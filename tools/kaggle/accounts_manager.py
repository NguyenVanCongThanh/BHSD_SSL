#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
accounts_manager.py: Quản lý tập trung các tài khoản Kaggle.
Hỗ trợ đa tài khoản (multi-account), tự động đồng bộ hóa credentials,
tạo isolated profile directory với quyền chmod 600 để tránh xung đột môi trường.
"""

import os
import sys
import json
import stat
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "configs" / "kaggle" / "accounts.json"
SECRETS_DIR = PROJECT_ROOT / "configs" / "kaggle" / "secrets"

class AccountsManager:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = Path(config_path) if config_path else CONFIG_FILE
        self.accounts_data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            default_structure = {
                "default_account": "1",
                "accounts": {}
            }
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(default_structure, f, indent=2, ensure_ascii=False)
            return default_structure
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_config(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.accounts_data, f, indent=2, ensure_ascii=False)

    def get_account(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Tìm tài khoản theo ID ('1', '2'), slug ('thanh', 'khanh'), hoặc username.
        """
        identifier = str(identifier).strip()
        accounts = self.accounts_data.get("accounts", {})
        
        # 1. Tìm chính xác theo account key
        if identifier in accounts:
            return accounts[identifier]
        
        # 2. Tìm theo slug hoặc username hoặc id
        for acc_id, acc in accounts.items():
            if str(acc.get("id")) == identifier:
                return acc
            if acc.get("slug", "").lower() == identifier.lower():
                return acc
            if acc.get("username", "").lower() == identifier.lower():
                return acc
                
        return None

    def get_default_account_id(self) -> str:
        return str(self.accounts_data.get("default_account", "1"))

    def resolve_credentials(self, account: Dict[str, Any]) -> Tuple[str, str]:
        """
        Trích xuất (username, key) cho tài khoản đã chọn.
        Hỗ trợ fallback sang ~/.kaggle/kaggle.json nếu được chỉ định.
        """
        username = account.get("username", "").strip()
        key = account.get("key", "").strip()
        source = account.get("source", "key")

        # Kiểm tra file toàn cục ~/.kaggle/kaggle.json nếu source là global
        if source == "global_or_key":
            global_path = Path.home() / ".kaggle" / "kaggle.json"
            if global_path.exists():
                try:
                    with open(global_path, "r", encoding="utf-8") as f:
                        gdata = json.load(f)
                    guser = gdata.get("username", "").strip()
                    gkey = gdata.get("key", "").strip()
                    if guser and gkey:
                        return guser, gkey
                except Exception:
                    pass

        if not username or not key:
            raise ValueError(f"Tài khoản '{account.get('name', 'Unknown')}' thiếu Username hoặc API Key hợp lệ!")
        return username, key

    def create_isolated_profile(self, account_id_or_slug: str) -> Path:
        """
        Tạo một profile directory biệt lập chứa kaggle.json an toàn (chmod 600)
        cho tài khoản được chọn, trả về đường dẫn dùng làm KAGGLE_CONFIG_DIR.
        """
        acc = self.get_account(account_id_or_slug)
        if not acc:
            raise KeyError(f"Không tìm thấy tài khoản với định danh: '{account_id_or_slug}'")

        username, key = self.resolve_credentials(acc)
        acc_key = str(acc.get("id", acc.get("slug", "default")))

        profile_dir = SECRETS_DIR / f"profile_{acc_key}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        # chmod 700 cho thư mục
        try:
            os.chmod(profile_dir, stat.S_IRWXU)
        except Exception:
            pass

        kaggle_json_path = profile_dir / "kaggle.json"
        with open(kaggle_json_path, "w", encoding="utf-8") as f:
            json.dump({"username": username, "key": key}, f, indent=2)

        # Chmod 600 theo đúng yêu cầu bảo mật của Kaggle CLI
        try:
            os.chmod(kaggle_json_path, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass

        return profile_dir

    def list_accounts(self) -> Dict[str, Any]:
        return self.accounts_data.get("accounts", {})

    def add_or_update_account(self,
                             acc_id: str,
                             username: str,
                             key: str,
                             name: Optional[str] = None,
                             slug: Optional[str] = None,
                             description: Optional[str] = None) -> Dict[str, Any]:
        """
        Thêm mới hoặc cập nhật thông tin tài khoản vào registry.
        """
        acc_id = str(acc_id).strip()
        existing = self.accounts_data.get("accounts", {}).get(acc_id, {})
        
        updated = {
            "id": acc_id,
            "slug": slug or existing.get("slug") or f"acc{acc_id}",
            "name": name or existing.get("name") or f"Tài khoản {acc_id} ({username})",
            "username": username.strip(),
            "source": "key",
            "key": key.strip(),
            "description": description or existing.get("description") or f"Tài khoản Kaggle {username}",
            "default_dataset_owner": existing.get("default_dataset_owner") or username.strip()
        }
        
        self.accounts_data.setdefault("accounts", {})[acc_id] = updated
        self._save_config()
        # Tạo lại profile isolated
        self.create_isolated_profile(acc_id)
        return updated
