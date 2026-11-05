"""
Legacy wrapper for per-class trend comparison.
Delegates to unified tools/compare_runs.py --mode classes.
"""
try:
    from .compare_runs import main
except ImportError:
    from compare_runs import main

if __name__ == "__main__":
    main(mode='classes')
