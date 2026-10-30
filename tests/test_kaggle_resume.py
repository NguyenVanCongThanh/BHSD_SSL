import ast
import copy
import json
import sys
import types
from pathlib import Path

import unittest
import tempfile
from unittest.mock import patch

from tools.generate_mean_teacher_notebook import build_mean_teacher_notebook
from tools.kaggle.resume import inject_resume, prepare_resume


def check_inject_resume_only_changes_training_cell():
    notebook = build_mean_teacher_notebook('binary')
    original = copy.deepcopy(notebook)
    inject_resume(notebook, 'klbm4ebr', 'best')
    changed = [i for i, (a, b) in enumerate(zip(original['cells'], notebook['cells'])) if a != b]
    assert len(changed) == 1
    source = ''.join(notebook['cells'][changed[0]]['source'])
    ast.parse(source)
    assert "prepare_resume('klbm4ebr', 'best', DATA_ROOT" in source
    assert source.index('base_args, resume_cfg') < source.index('train_cmd =')
    assert 'prepare_resume' not in json.dumps(original)


def check_prepare_resume_restores_source_config_and_owned_checkpoint(tmp_path):
    config = {'experiment': {'name': 'original', 'fold': 0, 'task': 'binary'},
              'method': {'name': 'mean_teacher', 'max_iteration': 12000},
              'data': {'root_path': 'old', 'batch_size': 8}, 'logging': {},
              'resolved': {'batch_layout': {'world_size': 2}}}
    class Artifact:
        name = 'original_fold0_best:v27'
        metadata = {'iter_num': 7000}
        def download(self, root):
            Path(root).mkdir(parents=True)
            (Path(root) / 'best_model.pth').write_bytes(b'checkpoint')
            return root
    run = types.SimpleNamespace(config=config, state='crashed', logged_artifacts=lambda: [Artifact()])
    sys.modules['wandb'] = types.SimpleNamespace(Api=lambda: types.SimpleNamespace(run=lambda p: run))
    args, cfg = prepare_resume('klbm4ebr', 'best', '/input/data', tmp_path, 2)
    assert '--resume' in args and '--ddp' in args
    assert cfg['method']['max_iteration'] == 12000
    assert cfg['logging']['wandb_id'] == 'klbm4ebr'
    assert cfg['data']['root_path'] == '/input/data'
    assert config['data']['root_path'] == 'old'
    assert (tmp_path / 'original/fold_0/checkpoint_latest.pth').read_bytes() == b'checkpoint'
    _, branch = prepare_resume('klbm4ebr', 'best', '/input/data', tmp_path, 2, 'new-run')
    assert branch['experiment']['name'].startswith('original_resume_')
    assert branch['experiment']['resume'] is True
    assert branch['logging']['wandb_id'] is None
    assert branch['logging']['wandb_resume_new_run'] is True
    assert branch['logging']['resume_source']['metadata']['iter_num'] == 7000
    assert (tmp_path / branch['experiment']['name'] / 'fold_0/checkpoint_latest.pth').read_bytes() == b'checkpoint'
    with unittest.TestCase().assertRaises(FileExistsError):
        prepare_resume('klbm4ebr', 'best', '/input/data', tmp_path, 2)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'world size'):
        prepare_resume('klbm4ebr', 'best', '/input/data', tmp_path, 1)
    with unittest.TestCase().assertRaises(FileNotFoundError):
        prepare_resume('klbm4ebr', 'latest', '/input/data', tmp_path, 2)


class ResumeTests(unittest.TestCase):
    def test_inject(self):
        check_inject_resume_only_changes_training_cell()

    def test_prepare(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules):
            check_prepare_resume_restores_source_config_and_owned_checkpoint(Path(directory))

    def test_builder_fresh_resume_then_fresh(self):
        from tools.kaggle import notebook_builder
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(notebook_builder, 'RUNS_BASE_DIR', Path(directory)), \
                patch.object(notebook_builder.subprocess, 'run'):
            builder = notebook_builder.NotebookBuilder()
            folder = builder.build_bundle('mean_teacher', 'test_user')
            meta = json.loads((folder / 'kernel-metadata.json').read_text())
            code = folder / meta['code_file']
            fresh = code.read_bytes()
            self.assertNotIn(b'def prepare_resume', fresh)
            builder.build_bundle('mean_teacher', 'test_user', resume_run='klbm4ebr', resume_checkpoint='best')
            self.assertIn(b'def prepare_resume', code.read_bytes())
            builder.build_bundle('mean_teacher', 'test_user', resume_run='klbm4ebr',
                                 resume_checkpoint='best', resume_mode='new-run')
            branch_nb = json.loads(code.read_text())
            branch_cell = next(''.join(c['source']) for c in branch_nb['cells']
                               if 'def prepare_resume' in ''.join(c.get('source', [])))
            ast.parse(branch_cell)
            self.assertIn('resume_mode=\'new-run\'', branch_cell)
            self.assertIn('Path("utils/wandb_manager.py").write_text', branch_cell)
            builder.build_bundle('mean_teacher', 'test_user')
            self.assertEqual(code.read_bytes(), fresh)
            with self.assertRaises(ValueError):
                builder.build_bundle('mean_teacher', 'test_user', resume_checkpoint='best')
            with self.assertRaises(ValueError):
                builder.build_bundle('pick', 'test_user', resume_run='klbm4ebr')


    def test_new_segment_initializes_fresh_tracking_identity(self):
        from utils.wandb_manager import WandbManager
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules):
            run = types.SimpleNamespace(id='new-id', entity='team', define_metric=Mock())
            sdk = types.SimpleNamespace(init=Mock(return_value=run))
            sys.modules['wandb'] = sdk
            manager = WandbManager.__new__(WandbManager)
            manager.cfg = types.SimpleNamespace(
                experiment=types.SimpleNamespace(resume=True, name='branch'),
                method=types.SimpleNamespace(name='mean_teacher'),
                logging=types.SimpleNamespace(wandb_resume_new_run=True, wandb_id=None))
            manager.cfg.to_dict = lambda: {}
            manager.run_name = None
            manager.fold = 0
            manager.project = 'project'
            manager.entity = 'team'
            manager.mode = 'online'
            manager.run = None
            manager.identity_path = Path(directory) / 'wandb_run.json'
            manager._resume_identity = Mock(side_effect=AssertionError('Must not inherit source ID'))
            manager._init_wandb()
            self.assertEqual(sdk.init.call_args.kwargs['resume'], 'never')
            self.assertNotIn('id', sdk.init.call_args.kwargs)
            self.assertEqual(json.loads(manager.identity_path.read_text())['id'], 'new-id')
