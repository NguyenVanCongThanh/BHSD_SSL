"""Checks for physical units, preprocessing loss detection and sampler fidelity."""
import tempfile
import unittest
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

from tools.bhsd_eda import morphology, profile_case, sample_case, split_audit
from data.transforms3d import RandomCrop


class BHSDExplorerTests(unittest.TestCase):
    def test_physical_units_and_scale(self):
        mask = np.zeros((7, 9, 11), bool)
        mask[2:5, 2:7, 3:8] = True
        a = morphology(mask, np.array([5., .5, .5]), 'case', 2)
        b = morphology(mask, np.array([10., 1., 1.]), 'case', 2)
        self.assertAlmostEqual(a['savr'], a['surface_area_cm2']/a['vol_ml'])
        self.assertAlmostEqual(b['savr'], a['savr']/2)
        self.assertAlmostEqual(b['sphericity'], a['sphericity'])
        self.assertEqual(b['components_26'], 1)
        single = morphology(np.ones((1, 1, 1), bool), np.ones(3), 'tiny', 2)
        self.assertEqual(single['voxels'], 1)
        self.assertFalse(single['pca_defined'])
        self.assertTrue(np.isnan(single['flatness']))

    def test_crop_loss_and_disappeared_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'data').mkdir()
            (root/'cache').mkdir()
            for name in ['images','ground truths']:
                (root/name).mkdir()
            native = np.zeros((4, 6, 6), np.uint8)
            native[1, 0, 0] = 1  # removed by crop
            native[1, 2, 2] = 2  # remains in crop, then disappears in H5
            image = np.zeros_like(native, dtype=np.float32)
            affine = np.diag([.5, .5, 5., 1.])
            nib.save(nib.Nifti1Image(native.transpose(2,1,0), affine), root/'ground truths/case.nii.gz')
            nib.save(nib.Nifti1Image(image.transpose(2,1,0), affine), root/'images/case.nii.gz')
            attrs = dict(target_spacing='(5., .5, .5)', orig_shape='(4,6,6)',
                orig_spacing='(5., .5, .5)', bbox_used='[[0,4],[1,5],[1,5]]',
                cropped_shape='(4,4,4)', final_shape='(4,4,4)', is_labeled=True,
                has_label='True', case_id='case', clip_range='[11.,88.]',
                norm_params="{'mean':47.6804,'std':18.4999}")
            with h5py.File(root/'data/case.h5','w') as f:
                f.create_dataset('image',data=np.zeros((4,4,4),np.float32))
                f.create_dataset('label',data=np.zeros((4,4,4),np.uint8))
                for key,value in attrs.items(): f.attrs[key] = value
            result = profile_case('case',root,root,root/'cache','train_labeled')
            self.assertEqual(result['summary']['raw_status'],'metadata_consistent')
            self.assertEqual(result['retention'][0]['cropped_out_voxels'],1)
            self.assertTrue(result['retention'][1]['disappeared'])
            self.assertEqual(result['retention'][1]['cropped_out_voxels'],0)

    def test_sampler_matches_production_with_padding_and_jitter(self):
        label = np.zeros((3, 10, 10), np.uint8)
        label[1, 3:5, 3:5] = 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'data').mkdir()
            with h5py.File(root/'data/case.h5','w') as f:
                f.create_dataset('label',data=label)
            np.random.seed(123)
            before = np.random.get_state()
            result = sample_case('case',root,trials=30,patch_sizes=((4,6,6),),probabilities=(.5,),seed=42)[0]
            after = np.random.get_state()
            np.testing.assert_array_equal(before[1],after[1])
            self.assertEqual(before[2],after[2])
            np.random.seed(42)
            crop = RandomCrop((4,6,6),foreground_prob=.5)
            hits = [np.any(crop({'image':label,'label':label})['label'] > 0) for _ in range(30)]
            self.assertEqual(result['any_hit_rate'],np.mean(hits))
            self.assertEqual(result['c1_hit_rate'],np.mean(hits))

    def test_split_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'data').mkdir()
            for k,v in {'train':'a a','val':'b','test':'b','train_labeled':'a','train_unlabeled':''}.items():
                (root/f'{k}.txt').write_text(v)
            for case in ['a','b']: (root/'data'/f'{case}.h5').touch()
            with self.assertRaisesRegex(AssertionError,'Duplicate'):
                split_audit(root)


if __name__ == '__main__':
    unittest.main()
