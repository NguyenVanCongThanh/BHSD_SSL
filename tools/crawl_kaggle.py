"""Save Kaggle launcher logs without downloading model checkpoints."""
import argparse
import json
import os
from pathlib import Path

try:
    from .crawl_storage import PROJECT_ROOT, new_snapshot, write_json
except ImportError:
    from crawl_storage import PROJECT_ROOT, new_snapshot, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kernel', help='owner/kernel_slug')
    parser.add_argument('--account', help='Account ID in configs/kaggle/accounts.json; otherwise use Kaggle credentials')
    args = parser.parse_args()
    if args.account:
        accounts = json.loads((PROJECT_ROOT / 'configs/kaggle/accounts.json').read_text())['accounts']
        account = accounts[args.account]
        key = account.get('key') or account.get('token')
        if not key:
            raise ValueError('Selected account has no key/token; configure Kaggle credentials.')
        for name in ('KAGGLE_USERNAME', 'KAGGLE_KEY', 'KAGGLE_API_TOKEN'):
            os.environ.pop(name, None)
        os.environ['KAGGLE_USERNAME'] = account['username']
        os.environ['KAGGLE_KEY'] = key
        if key.startswith('KGAT_'):
            os.environ['KAGGLE_API_TOKEN'] = key
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
    api = KaggleApi()
    api.authenticate()
    owner, slug = args.kernel.split('/')
    folder = new_snapshot('kaggle', args.kernel, kernel=args.kernel,
                          version=None, version_note='API returns latest session output; version ID not supplied')
    status = api.kernels_status(args.kernel)
    write_json(folder / 'status.json', json.loads(str(status)))
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner
        request.kernel_slug = slug
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
    raw = response.log or ''
    (folder / 'output.log.json').write_text(raw, encoding='utf-8')
    # Save names, never expiring signed download URLs or credentials.
    write_json(folder / 'files.json', [{'name': f.file_name} for f in response.files])
    print(folder)


if __name__ == '__main__':
    main()
