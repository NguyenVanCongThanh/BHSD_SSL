#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kaggle_ctl.py: CLI điều khiển trung tâm quản lý tài khoản & submit Kaggle Notebook.
Cung cấp giao diện trực quan, chuyên nghiệp và mở rộng không giới hạn cho mọi
tài khoản và mọi notebook sau này.
"""

import sys
import json
import ast
import re
import argparse
from pathlib import Path

# Đảm bảo import được module local
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.kaggle.accounts_manager import AccountsManager
from tools.kaggle.notebook_builder import NotebookBuilder
from tools.kaggle.kaggle_executor import KaggleExecutor

def cmd_account_list(args):
    mgr = AccountsManager()
    accounts = mgr.list_accounts()
    print("\n" + "=" * 80)
    print("DANH SÁCH TÀI KHOẢN KAGGLE ĐƯỢC QUẢN LÝ (MedSSL-BHSD)")
    print("=" * 80)
    print(f"{'ID':<4} | {'Slug':<10} | {'Username':<18} | {'Credential Status':<18} | {'Tên hiển thị'}")
    print("-" * 80)
    for acc_id, acc in accounts.items():
        key = (acc.get("key") or "").strip()
        status_str = "Configured" if key else "(from env/global)"
        print(f"{acc_id:<4} | {acc.get('slug', ''):<10} | {acc.get('username', ''):<18} | {status_str:<18} | {acc.get('name', '')}")
    print("=" * 80 + "\n")

def cmd_account_test(args):
    mgr = AccountsManager()
    target_ids = [args.account] if args.account else list(mgr.list_accounts().keys())
    print("\n" + "=" * 80)
    print("KIỂM TRA KẾT NỐI TÀI KHOẢN KAGGLE API")
    print("=" * 80)
    for acc_id in target_ids:
        acc = mgr.get_account(acc_id)
        if not acc:
            print(f"[!] Không tìm thấy tài khoản với id: {acc_id}")
            continue
        print(f"[*] Kiểm tra: {acc.get('name')} (User: {acc.get('username')})...", end=" ", flush=True)
        try:
            profile_dir = mgr.create_isolated_profile(acc_id)
            executor = KaggleExecutor(profile_dir)
            ok, msg = executor.test_connection()
            if ok:
                print(f"[✓ HỢP LỆ]")
            else:
                print(f"[X THẤT BẠI] ({msg})")
        except Exception as e:
            print(f"[X LỖI] {e}")
    print("=" * 80 + "\n")

def cmd_account_add(args):
    mgr = AccountsManager()
    updated = mgr.add_or_update_account(
        acc_id=args.id,
        username=args.username,
        key=args.key,
        name=args.name,
        slug=args.slug,
        description=args.desc
    )
    print(f"\n[✓] Đã lưu thành công tài khoản ID '{args.id}' (User: {updated['username']}) vào registry.")
    print(f"    Profile cô lập đã được tạo sẵn sàng.\n")

def cmd_push(args):
    mgr = AccountsManager()
    builder = NotebookBuilder()

    account_id = args.account or mgr.get_default_account_id()
    acc = mgr.get_account(account_id)
    if not acc:
        print(f"[X] Lỗi: Không tìm thấy tài khoản '{account_id}' trong danh sách!")
        sys.exit(1)

    username, _ = mgr.resolve_credentials(acc)
    profile_dir = mgr.create_isolated_profile(account_id)
    executor = KaggleExecutor(profile_dir)

    notebook_type = args.notebook or "mean_teacher"
    print("\n" + "=" * 80)
    print(f"SUBMIT NOTEBOOK '{notebook_type.upper()}' LÊN KAGGLE")
    print(f"Tài khoản đích: {acc.get('name')} (Username: {username})")
    print("=" * 80)

    # Đóng gói notebook và kernel-metadata.json tự động tạo mới theo username
    custom_datasets = [args.dataset] if args.dataset else None
    bundle_dir = builder.build_bundle(
        notebook_type=notebook_type,
        target_username=username,
        custom_slug=args.slug,
        custom_title=args.title,
        custom_datasets=custom_datasets,
        enable_gpu=not args.no_gpu
    )

    print(f"[*] Thư mục bundle đã chuẩn bị: {bundle_dir}")
    with open(bundle_dir / "kernel-metadata.json", "r", encoding="utf-8") as f:
        meta_info = json.load(f)
        print("    Kernel ID:", meta_info.get("id"))

    # Thực hiện push
    result = executor.push_kernel(bundle_dir)

    print("\n" + "=" * 80)
    if result.get("success"):
        print(f"[✓ THÀNH CÔNG] Notebook đã được đẩy lên Kaggle cho tài khoản {username}!")
        print(f"[*] Kernel ID: {result.get('kernel_id')}")
        print(f"[*] Link trực tiếp: {result.get('url')}")
        print(f"\n[*] Để kiểm tra tiến trình chạy, dùng lệnh:")
        print(f"    python3 tools/kaggle/kaggle_ctl.py status --account {account_id} --kernel {result.get('kernel_id')}")
    else:
        print(f"[X THẤT BẠI] Quá trình push gặp lỗi:")
        print(result.get("error") or result.get("message"))
        print("\n--> Gợi ý: Hãy kiểm tra tính hợp lệ của API key hoặc quyền truy cập dataset nguồn.")
    print("=" * 80 + "\n")

def cmd_status(args):
    mgr = AccountsManager()
    account_id = args.account or mgr.get_default_account_id()
    acc = mgr.get_account(account_id)
    if not acc:
        print(f"[X] Không tìm thấy tài khoản '{account_id}'")
        sys.exit(1)
        
    username, _ = mgr.resolve_credentials(acc)
    profile_dir = mgr.create_isolated_profile(account_id)
    executor = KaggleExecutor(profile_dir)

    kernel_id = args.kernel
    if not kernel_id:
        notebook_type = args.notebook or "mean_teacher"
        builder = NotebookBuilder()
        spec = builder.get_notebook_spec(notebook_type)
        slug = spec.get("slug", "kaggle-bhsd-mean-teacher-dual-gpu")
        kernel_id = f"{username}/{slug}"

    print(f"[*] Đang kiểm tra trạng thái kernel: {kernel_id}...")
    status = executor.get_status(kernel_id)
    print(f"[>] Trạng thái: {status}")

def cmd_log(args):
    mgr = AccountsManager()
    acc = mgr.get_account(args.account)
    if not acc:
        print(f"[X] Không tìm thấy tài khoản với định danh: {args.account}")
        sys.exit(1)
    account_id = acc.get("id", args.account)
    username, _ = mgr.resolve_credentials(acc)
    profile_dir = mgr.create_isolated_profile(account_id)
    executor = KaggleExecutor(profile_dir)

    kernel_id = args.kernel
    if not kernel_id:
        notebook_type = args.notebook or "mean_teacher"
        builder = NotebookBuilder()
        spec = builder.get_notebook_spec(notebook_type)
        slug = spec.get("slug", "kaggle-bhsd-mean-teacher-dual-gpu")
        kernel_id = f"{username}/{slug}"
    else:
        slug = kernel_id.split("/")[-1]

    print(f"[*] Đang kéo log từ Kaggle cho kernel: {kernel_id}...")
    ok, content = executor.get_log(kernel_id)
    if not ok:
        print(f"[X] {content}")
        sys.exit(1)

    if args.output:
        out_path = Path(args.output)
    else:
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        out_path = logs_dir / f"{slug}.log"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    lines = content.splitlines()
    print(f"[✓] Đã lưu log thành công: {out_path} ({len(lines):,} dòng, {len(content):,} bytes)")

    tail_n = args.tail if args.tail is not None else 25
    if tail_n > 0 and lines:
        print(f"\n--- {min(tail_n, len(lines))} DÒNG CUỐI CỦA LOG ---")
        for line in lines[-tail_n:]:
            print(line)
        print("=" * 60)

def cmd_build(args):
    builder = NotebookBuilder()
    notebook_type = args.notebook or "mean_teacher"
    username = args.username or "dry_run_user"
    print("\n" + "=" * 80)
    print(f"BUILD & VALIDATE NOTEBOOK '{notebook_type.upper()}' CỤC BỘ (DRY-RUN)")
    print(f"Target Username: {username}")
    print("=" * 80)

    bundle_dir = builder.build_bundle(
        notebook_type=notebook_type,
        target_username=username,
        custom_slug=args.slug,
        custom_title=args.title,
        custom_datasets=[args.dataset] if args.dataset else None,
        enable_gpu=not args.no_gpu
    )

    metadata_file = bundle_dir / "kernel-metadata.json"
    if not metadata_file.exists():
        print(f"[X] Lỗi: kernel-metadata.json không tồn tại trong {bundle_dir}")
        sys.exit(1)

    with open(metadata_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required_keys = ["id", "code_file", "language", "kernel_type", "is_private", "enable_gpu"]
    for k in required_keys:
        if k not in meta:
            print(f"[X] Metadata thiếu key bắt buộc: {k}")
            sys.exit(1)

    code_path = bundle_dir / meta["code_file"]
    if not code_path.exists():
        print(f"[X] File notebook không tồn tại: {code_path}")
        sys.exit(1)

    with open(code_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        if "cells" not in nb_data:
            print("[X] Notebook JSON không hợp lệ (thiếu cells)")
            sys.exit(1)

    # Validate Python code syntax and secrets in all code cells
    secret_patterns = [r"wandb_v1_[A-Za-z0-9_]{50,}", r"KGAT_[a-f0-9]{32}"]
    for idx, cell in enumerate(nb_data.get("cells", [])):
        if cell.get("cell_type") == "code":
            raw_src = cell.get("source", "")
            source = "".join(raw_src) if isinstance(raw_src, list) else str(raw_src)
            # Check for embedded credentials
            for pat in secret_patterns:
                if re.search(pat, source):
                    print(f"[X] Phát hiện token nhạy cảm trong cell code #{idx}!")
                    sys.exit(1)
            # Filter IPython magics and shell commands (!pip, %cd, %%writefile, etc.)
            clean_lines = []
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith(('!', '%')):
                    clean_lines.append("# " + line)
                else:
                    clean_lines.append(line)
            py_code = "\n".join(clean_lines)
            try:
                ast.parse(py_code)
            except SyntaxError as e:
                print(f"[X] Lỗi cú pháp Python trong cell code #{idx}: {e}")
                sys.exit(1)

    print(f"[✓ HỢP LỆ] Bundle xây dựng thành công tại: {bundle_dir}")
    print(f"    Kernel ID:   {meta.get('id')}")
    print(f"    Code File:   {meta.get('code_file')} ({len(nb_data.get('cells', []))} cells)")
    print(f"    GPU Enabled: {meta.get('enable_gpu')}")
    print("=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="MedSSL-BHSD Kaggle Controller: Quản lý đa tài khoản & submit notebook Kaggle chuyên nghiệp",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Lệnh cần thực hiện")

    # Subcommand: account
    parser_acc = subparsers.add_parser("account", help="Quản lý tài khoản Kaggle")
    sub_acc = parser_acc.add_subparsers(dest="acc_command")
    
    sub_acc.add_parser("list", help="Liệt kê danh sách tài khoản")
    
    test_acc = sub_acc.add_parser("test", help="Kiểm tra kết nối và tính hợp lệ của tài khoản")
    test_acc.add_argument("--account", "-a", help="ID hoặc slug tài khoản cụ thể cần test")
    
    add_acc = sub_acc.add_parser("add", help="Thêm mới hoặc cập nhật tài khoản")
    add_acc.add_argument("--id", required=True, help="ID tài khoản (ví dụ: 3)")
    add_acc.add_argument("--username", "-u", required=True, help="Kaggle username")
    add_acc.add_argument("--key", "-k", required=True, help="Kaggle API Key (KGAT_...)")
    add_acc.add_argument("--name", "-n", help="Tên hiển thị dễ nhớ")
    add_acc.add_argument("--slug", "-s", help="Slug viết tắt (ví dụ: user3)")
    add_acc.add_argument("--desc", "-d", help="Mô tả bổ sung")

    # Subcommand: build
    parser_build = subparsers.add_parser("build", help="Xây dựng và kiểm tra tính hợp lệ của bundle notebook cục bộ (không gọi mạng)")
    parser_build.add_argument("--notebook", "-n", default="mean_teacher", help="Loại notebook ('mean_teacher', 'pick')")
    parser_build.add_argument("--username", "-u", default="dry_run_user", help="Username giả định để build metadata")
    parser_build.add_argument("--slug", help="Ghi đè slug của kernel")
    parser_build.add_argument("--title", help="Ghi đè title của notebook")
    parser_build.add_argument("--dataset", "-d", help="Ghi đè dataset source")
    parser_build.add_argument("--no-gpu", action="store_true", help="Vô hiệu hóa GPU")

    # Subcommand: push
    parser_push = subparsers.add_parser("push", help="Đẩy notebook lên Kaggle")
    parser_push.add_argument("--account", "-a", default="1", help="Tài khoản để submit (ID '1', '2' hoặc slug 'thanh', 'khanh')")
    parser_push.add_argument("--notebook", "-n", default="mean_teacher", help="Loại notebook ('mean_teacher', 'pick')")
    parser_push.add_argument("--slug", help="Ghi đè slug của kernel trên Kaggle")
    parser_push.add_argument("--title", help="Ghi đè title của notebook")
    parser_push.add_argument("--dataset", "-d", help="Ghi đè dataset source (mặc định thnhnguynvncng/bhsd-fold0)")
    parser_push.add_argument("--no-gpu", action="store_true", help="Vô hiệu hóa GPU")

    # Subcommand: status
    parser_status = subparsers.add_parser("status", help="Kiểm tra trạng thái notebook đang chạy")
    parser_status.add_argument("--account", "-a", default="1", help="Tài khoản thực hiện kiểm tra")
    parser_status.add_argument("--notebook", "-n", default="mean_teacher", help="Loại notebook")
    parser_status.add_argument("--kernel", "-k", help="Định danh kernel đầy đủ (username/slug)")

    # Subcommand: log
    parser_log = subparsers.add_parser("log", help="Kéo riêng file log văn bản từ Kaggle về (nhẹ, nhanh, không tải checkpoint)")
    parser_log.add_argument("--account", "-a", default="1", help="Tài khoản (ID hoặc slug, mặc định: 1)")
    parser_log.add_argument("--notebook", "-n", default="mean_teacher", help="Loại notebook ('mean_teacher', 'pick')")
    parser_log.add_argument("--kernel", "-k", help="Định danh kernel đầy đủ (username/slug)")
    parser_log.add_argument("--output", "-o", help="Đường dẫn file đích để lưu log (mặc định: logs/<kernel_slug>.log)")
    parser_log.add_argument("--tail", "-t", type=int, default=25, help="Số dòng cuối cần in ra màn hình sau khi tải (mặc định: 25, 0 để tắt)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "account":
        if args.acc_command == "list" or not args.acc_command:
            cmd_account_list(args)
        elif args.acc_command == "test":
            cmd_account_test(args)
        elif args.acc_command == "add":
            cmd_account_add(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "push":
        cmd_push(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "log":
        cmd_log(args)

if __name__ == "__main__":
    main()
