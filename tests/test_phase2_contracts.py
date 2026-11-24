"""Phase 2 contract tests: Kaggle renderer unification, resume CLI & provenance, and core/constants cycle-breaking."""
import ast
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestCoreConstantsAndImportIsolation(unittest.TestCase):
    def test_core_package_has_zero_imports_from_data(self):
        core_dir = REPO_ROOT / "core"
        offenders = []
        for py_file in sorted(core_dir.glob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "data" or alias.name.startswith("data."):
                            offenders.append(f"{py_file.name}:{node.lineno} import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == "data" or node.module.startswith("data."):
                        offenders.append(f"{py_file.name}:{node.lineno} from {node.module}")
        self.assertEqual(offenders, [], f"Circular core -> data imports remain: {offenders}")

    def test_lightweight_imports_do_not_pull_heavy_runtime_modules(self):
        script = (
            "import sys\n"
            "import core.constants\n"
            "import core.config_parser\n"
            "import core.model_builder\n"
            "import data.constants\n"
            "import tools.setup_wandb_dashboard\n"
            "loaded = [m for m in ('h5py', 'scipy', 'PIL', 'torchvision') if m in sys.modules]\n"
            "assert not loaded, f'Heavy modules unexpectedly loaded: {loaded}'\n"
            "assert data.constants.DEFAULT_PATCH_SIZE == core.constants.DEFAULT_PATCH_SIZE\n"
        )
        res = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, f"Isolation check failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")

    def test_lazy_data_exports_preserve_backward_compatibility(self):
        import data
        self.assertTrue(hasattr(data, "BHSD_MULTICLASS_NAMES"))
        self.assertTrue(hasattr(data, "BHSDDataset"))
        self.assertIn("BHSDDataset", dir(data))


class TestUnifiedKaggleRenderer(unittest.TestCase):
    def test_all_four_methods_forward_cli_overrides_uniformly(self):
        from tools.kaggle.render_notebook import NotebookRenderOptions, render_training_notebook

        opts = NotebookRenderOptions(
            task="binary",
            max_iteration=7777,
            base_lr=0.005,
            batch_size=6,
            labeled_bs=3,
            single_gpu=True,
            run_name="phase2_uniform_check",
            extra_overrides=[("method.consistency_rampup_unit", "iter")],
        )
        for method_key in ("supervised", "mean_teacher", "lefed", "pick"):
            with self.subTest(method=method_key):
                nb = render_training_notebook(method_key, options=opts)
                code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
                all_code = "\n".join(code_cells)
                for cell_src in code_cells:
                    cleaned = "\n".join(
                        (ln[: len(ln) - len(ln.lstrip())] + "pass") if ln.lstrip().startswith(("!", "%")) else ln
                        for ln in cell_src.splitlines()
                    )
                    ast.parse(cleaned)
                self.assertIn('TASK = "binary"', all_code)
                self.assertIn("MAX_ITER = 7777", all_code)
                self.assertIn("BATCH_SIZE = 6", all_code)
                self.assertIn("LABELED_BS = 3", all_code)
                self.assertIn("FORCE_SINGLE_GPU = True", all_code)
                self.assertIn('EXP_NAME = "phase2_uniform_check"', all_code)
                self.assertIn('"--base_lr", "0.005"', all_code)
                self.assertIn('"--method.consistency_rampup_unit", "iter"', all_code)
                self.assertIn("# Cấu hình Anti-Deadlock", all_code)


class TestResumeCLIAndProvenance(unittest.TestCase):
    def test_inject_resume_default_clean_vs_opt_in_local_patches(self):
        from tools.generate_mean_teacher_notebook import build_mean_teacher_notebook
        from tools.kaggle.resume import inject_resume

        nb_clean = build_mean_teacher_notebook("binary")
        inject_resume(nb_clean, "run123", "best", "same-run")
        clean_src = next(
            "".join(c["source"]) for c in nb_clean["cells"]
            if c.get("cell_type") == "code" and "base_args = [" in "".join(c.get("source", []))
        )
        ast.parse(clean_src)
        self.assertIn("from tools.kaggle.resume import prepare_resume", clean_src)
        self.assertIn("local_patches_embedded=False", clean_src)
        self.assertNotIn("Path('core/config_parser.py').write_text", clean_src)
        self.assertNotIn("def prepare_resume", clean_src)

        nb_patched = build_mean_teacher_notebook("binary")
        inject_resume(nb_patched, "run123", "best", "same-run", embed_local_patches=True)
        patched_src = next(
            "".join(c["source"]) for c in nb_patched["cells"]
            if c.get("cell_type") == "code" and "base_args = [" in "".join(c.get("source", []))
        )
        ast.parse(patched_src)
        self.assertIn("local_patches_embedded=True", patched_src)
        self.assertIn("Path('core/config_parser.py').write_text", patched_src)

    def test_resume_cli_entrypoint_and_provenance_metadata(self):
        from tools.kaggle.resume import main as resume_main

        config = {
            "experiment": {"name": "mt_cli", "fold": 0, "task": "binary", "num_classes": 2},
            "method": {"name": "mean_teacher", "max_iteration": 12000},
            "data": {
                "name": "bhsd", "task": "binary", "num_classes": 2,
                "class_names": ["Background", "Hemorrhage"], "root_path": "old", "batch_size": 4,
            },
            "logging": {},
            "model": {"name": "vnet"},
            "resolved": {"batch_layout": {"world_size": 2}},
        }

        class _Art:
            name = "mt_cli_fold0_best:v1"
            metadata = {"iter_num": 3000}

            def download(self, root):
                Path(root).mkdir(parents=True)
                (Path(root) / "best_model.pth").write_bytes(b"ckpt")
                return root

        run = types.SimpleNamespace(config=config, state="finished", logged_artifacts=lambda: [_Art()])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules):
            sys.modules["wandb"] = types.SimpleNamespace(Api=lambda: types.SimpleNamespace(run=lambda p: run))
            out_clean = Path(tmp) / "out_clean.json"
            rc = resume_main([
                "--run", "ent/BHSD_MeanTeacher/r1",
                "--checkpoint", "best",
                "--data-root", "/data",
                "--exp-dir", str(Path(tmp) / "exp_clean"),
                "--num-gpus", "2",
                "--output-json", str(out_clean),
            ])
            self.assertEqual(rc, 0)
            payload_clean = json.loads(out_clean.read_text(encoding="utf-8"))
            self.assertFalse(payload_clean["logging"]["resume_source"]["local_patches_embedded"])
            self.assertEqual(payload_clean["logging"]["resume_source"]["code_provenance"], "git_commit")

            out_patched = Path(tmp) / "out_patched.json"
            rc2 = resume_main([
                "--run", "ent/BHSD_MeanTeacher/r1",
                "--checkpoint", "best",
                "--data-root", "/data",
                "--exp-dir", str(Path(tmp) / "exp_patched"),
                "--num-gpus", "2",
                "--local-patches-embedded",
                "--output-json", str(out_patched),
            ])
            self.assertEqual(rc2, 0)
            payload_patched = json.loads(out_patched.read_text(encoding="utf-8"))
            self.assertTrue(payload_patched["logging"]["resume_source"]["local_patches_embedded"])
            self.assertEqual(payload_patched["logging"]["resume_source"]["code_provenance"], "local_patches_embedded")


if __name__ == "__main__":
    unittest.main()
