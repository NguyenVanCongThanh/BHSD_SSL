#!/usr/bin/env python3
"""
Adapter for BHSD Binary PICK Training (2 classes: Background vs Hemorrhage).
Seamlessly redirects to the MedSSL-BHSD unified framework (train.py / methods.pick).
Full backward compatibility: preserves all legacy arguments (--root_path, --exp, --fold, --model, etc.).
"""

import sys
from train import main

if __name__ == "__main__":
    # Insert default config if not specified
    if '--config' not in sys.argv:
        sys.argv.insert(1, '--config')
        sys.argv.insert(2, 'configs/methods/pick.yaml')
    if '--task' not in sys.argv:
        sys.argv.extend(['--task', 'binary'])
    main()
