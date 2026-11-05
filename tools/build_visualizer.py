#!/usr/bin/env python3
"""
Build & Bundle Script for the MedSSL-BHSD Research Method Visualization System.
Extracts live configurations, source code snippets, and empirical run telemetry
from the repository, combines them with declarative method/backbone schemas and
the modular JS visualization engine, and outputs:
  1. docs/visualizer/index.html (Standalone interactive web application)
  2. docs/visualizer/medssl_bundle.json (Structured method-agnostic data bundle)
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualizer.bridge import (
    build_config_presets,
    collect_code_snippets,
    load_empirical_runs,
)
from visualizer.schemas.backbones import get_backbone_specs
from visualizer.schemas.methods import get_method_specs


def build_data_bundle() -> dict:
    return {
        "meta": {
            "system": "MedSSL-BHSD Research Method Visualization & Interactive Learning System",
            "version": "1.0.0",
            "primary_method": "mean_teacher",
        },
        "configs": build_config_presets(),
        "backbones": get_backbone_specs(),
        "methods": get_method_specs(),
        "snippets": collect_code_snippets(),
        "runs": load_empirical_runs(),
    }


def compile_html(bundle: dict) -> str:
    engine_js = (REPO_ROOT / "visualizer" / "engine" / "core_engine.js").read_text(encoding="utf-8")
    views_js = (REPO_ROOT / "visualizer" / "ui" / "views.js").read_text(encoding="utf-8")
    bundle_json = json.dumps(bundle, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MedSSL-BHSD — Research Method Visualization & Interactive Learning System</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    :root {{
      --background: #090d16;
      --card: #111827;
      --border: #1e293b;
      --foreground: #f8fafc;
      --muted-foreground: #94a3b8;
    }}
  </style>
</head>
<body class="bg-[#090d16] text-[#f8fafc] antialiased p-4">
  <div id="medssl-app-root" class="max-w-[1600px] mx-auto"></div>
  <script>
    window.MEDSSL_BUNDLE = {bundle_json};
  </script>
  <script>
{engine_js}
  </script>
  <script>
{views_js}
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Build MedSSL-BHSD Interactive Research Method Visualizer")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "docs" / "visualizer"),
        help="Output directory for index.html and medssl_bundle.json",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_data_bundle()
    bundle_path = out_dir / "medssl_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")

    html_content = compile_html(bundle)
    html_path = out_dir / "index.html"
    html_path.write_text(html_content, encoding="utf-8")

    print(f"[Visualizer] Exported data bundle: {bundle_path} ({bundle_path.stat().st_size:,} bytes)")
    print(f"[Visualizer] Compiled interactive app: {html_path} ({html_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

