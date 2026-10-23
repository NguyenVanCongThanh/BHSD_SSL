#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
notebook_builder.py: Chuẩn bị và đóng gói notebook cùng kernel-metadata.json
chuẩn hóa cho từng tài khoản và từng phương pháp (Mean Teacher, PICK, ...).
Đảm bảo giải quyết triệt để vấn đề: Tự động khởi tạo notebook trên tài khoản mới
nếu tài khoản đó chưa từng có notebook này trên Kaggle.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_CONFIG_FILE = PROJECT_ROOT / "configs" / "kaggle" / "notebooks.json"
RUNS_BASE_DIR = PROJECT_ROOT / "kaggle_runs"

class NotebookBuilder:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = Path(config_path) if config_path else NOTEBOOKS_CONFIG_FILE
        self.notebooks_data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {"default_notebook": "mean_teacher", "notebooks": {}}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_notebook_spec(self, notebook_type: str) -> Dict[str, Any]:
        specs = self.notebooks_data.get("notebooks", {})
        if notebook_type not in specs:
            # Tìm gần đúng
            for k, v in specs.items():
                if notebook_type.lower() in k.lower() or notebook_type.lower() in v.get("slug", "").lower():
                    return v
            raise KeyError(f"Không tìm thấy cấu hình notebook loại: '{notebook_type}'. Các loại hỗ trợ: {list(specs.keys())}")
        return specs[notebook_type]

    def build_bundle(self,
                     notebook_type: str,
                     target_username: str,
                     custom_slug: Optional[str] = None,
                     custom_title: Optional[str] = None,
                     custom_datasets: Optional[List[str]] = None,
                     enable_gpu: Optional[bool] = None) -> Path:
        """
        Chuẩn bị thư mục bundle hoàn chỉnh gồm file .ipynb và file kernel-metadata.json
        cho tài khoản target_username.
        """
        spec = self.get_notebook_spec(notebook_type)
        slug = custom_slug or spec.get("slug", f"bhsd-{notebook_type}")
        title = custom_title or spec.get("title", slug.replace("-", "_"))
        code_file_name = spec.get("code_file", f"{slug}.ipynb")

        # Thư mục bundle cô lập cho tài khoản và notebook này
        bundle_dir = RUNS_BASE_DIR / notebook_type / target_username
        bundle_dir.mkdir(parents=True, exist_ok=True)

        target_code_path = bundle_dir / code_file_name

        # 1. Tìm hoặc sinh file notebook
        possible_sources = [
            PROJECT_ROOT / "scripts" / code_file_name,
            PROJECT_ROOT / f"kaggle_{notebook_type}" / code_file_name,
            PROJECT_ROOT / code_file_name
        ]
        
        found_source = None
        for p in possible_sources:
            if p.exists():
                found_source = p
                break

        # Nếu chưa tìm thấy file và có generator script, chạy generator script
        if not found_source:
            gen_script = spec.get("generator_script")
            if gen_script:
                gen_path = PROJECT_ROOT / gen_script
                if gen_path.exists():
                    try:
                        subprocess.run([sys.executable, str(gen_path)], cwd=str(PROJECT_ROOT), check=True, timeout=30)
                        for p in possible_sources:
                            if p.exists():
                                found_source = p
                                break
                    except Exception as e:
                        print(f"[!] Cảnh báo khi chạy generator script {gen_script}: {e}")

        # Đồng bộ file notebook vào bundle
        if found_source and found_source.exists():
            shutil.copy2(found_source, target_code_path)
        elif not target_code_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file notebook nguồn {code_file_name} trong dự án!")


        # 2. Xử lý dataset sources
        datasets = custom_datasets if custom_datasets is not None else spec.get("dataset_sources", [])
        resolved_datasets = []
        for ds in datasets:
            # Thay thế biến {username} nếu có
            resolved_ds = ds.replace("{username}", target_username)
            resolved_datasets.append(resolved_ds)

        # 3. Tạo file kernel-metadata.json chuẩn xác cho Kaggle API
        # id dạng: <username>/<kernel_slug>
        kernel_id = f"{target_username}/{slug}"
        metadata = {
            "id": kernel_id,
            "title": title,
            "code_file": code_file_name,
            "language": spec.get("language", "python"),
            "kernel_type": spec.get("kernel_type", "notebook"),
            "is_private": spec.get("is_private", True),
            "enable_gpu": spec.get("enable_gpu", True) if enable_gpu is None else enable_gpu,
            "enable_tpu": spec.get("enable_tpu", False),
            "enable_internet": spec.get("enable_internet", True),
            "keywords": spec.get("keywords", []),
            "dataset_sources": resolved_datasets,
            "kernel_sources": spec.get("kernel_sources", []),
            "competition_sources": spec.get("competition_sources", []),
            "model_sources": spec.get("model_sources", [])
        }

        metadata_path = bundle_dir / "kernel-metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return bundle_dir
