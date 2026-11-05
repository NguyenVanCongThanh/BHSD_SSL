"""Backward-compatible shim redirecting to tools/archive/fetch_wandb.py."""
import sys
from pathlib import Path

# Add archive directory to path if needed and re-export
from tools.archive.fetch_wandb import *
from tools.archive.fetch_wandb import extract_metrics

