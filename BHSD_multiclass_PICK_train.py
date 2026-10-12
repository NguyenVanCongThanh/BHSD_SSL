#!/usr/bin/env python3
"""
Adapter for BHSD Multi-class PICK Training (6 classes).
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
    main()
