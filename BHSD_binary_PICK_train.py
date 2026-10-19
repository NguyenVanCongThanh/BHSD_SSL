#!/usr/bin/env python3
"""
Adapter for BHSD Binary PICK Training (2 classes: Background vs Hemorrhage).
Seamlessly redirects to the MedSSL unified benchmark framework (train.py).
Full backward compatibility: preserves all legacy arguments (--root_path, --exp, --fold, --model, etc.).
"""

import sys
from train import main

if __name__ == "__main__":
    if '--dataset' not in sys.argv and '--config' not in sys.argv:
        sys.argv.insert(1, '--dataset')
        sys.argv.insert(2, 'bhsd_binary')
    if '--method' not in sys.argv and '--config' not in sys.argv:
        sys.argv.insert(3, '--method')
        sys.argv.insert(4, 'pick')
    main()
