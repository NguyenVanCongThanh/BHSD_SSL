#!/usr/bin/env python3
"""
Adapter for BHSD Binary PICK Training (2 classes: Background vs Hemorrhage).
Seamlessly redirects to the MedSSL unified benchmark framework (train.py).
Full backward compatibility: preserves all legacy arguments (--root_path, --exp, --fold, --model, etc.).
"""

import sys
from train import main

if __name__ == "__main__":
    has_cfg = any(a in ('--config', '-c') or a.startswith('--config=') for a in sys.argv[1:])
    has_ds = any(a in ('--dataset', '--data.name') or a.startswith(('--dataset=', '--data.name=')) for a in sys.argv[1:])
    has_m = any(a in ('--method', '--method.name') or a.startswith(('--method=', '--method.name=')) for a in sys.argv[1:])
    prepend = []
    if not has_ds and not has_cfg:
        prepend += ['--dataset', 'bhsd_binary']
    if not has_m and not has_cfg:
        prepend += ['--method', 'pick']
    sys.argv[1:1] = prepend
    main()
