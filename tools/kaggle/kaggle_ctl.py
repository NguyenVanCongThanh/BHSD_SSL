#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kaggle_ctl.py: CLI điều khiển trung tâm quản lý tài khoản & submit Kaggle Notebook.
Cung cấp giao diện trực quan, chuyên nghiệp và mở rộng không giới hạn cho mọi
tài khoản và mọi notebook sau này.
"""

import sys
import json
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
    print(f"{'ID':<4} | {'Slug':<10} | {'Username':<18} | {'Key Prefix':<12} | {'Tên hiển thị'}")
    print("-" * 80)
    for acc_id, acc in accounts.items():
        key = acc.get("key", "")
        masked_key = (key[:8] + "...") if len(key) > 8 else (key or "(from global)")
        print(f"{acc_id:<4} | {acc.get('slug', ''):<10} | {acc.get('username', ''):<18} | {masked_key:<12} | {acc.get('name', '')}")
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
    elif args.command == "push":
        cmd_push(args)
    elif args.command == "status":
        cmd_status(args)

if __name__ == "__main__":
    main()
