#!/usr/bin/env python3
"""
Backward-compatible proxy for evaluate_5folds_benchmark.py.
Redirects to the unified evaluation engine in evaluate.py.
"""

import sys
from evaluate import main

if __name__ == "__main__":
    main()
