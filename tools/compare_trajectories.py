"""
Legacy wrapper for trajectory comparison.
Delegates to unified tools/compare_runs.py --mode trajectory.
"""
try:
    from .compare_runs import main
except ImportError:
    from compare_runs import main

if __name__ == "__main__":
    main(mode='trajectory')
