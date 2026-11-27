import torch
import pytest

from core.model_builder import model_spec_from_config, build_model_from_spec
from models.backbones.vnet import VNet
from models.outputs import extract_multiscale_outputs


def test_vnet5_default_backward_compatible():
    x = torch.randn(2, 1, 16, 64, 64)
    model = VNet(
        in_channels=1,
        num_classes=2,
        n_filters=16,
        num_stages=5,
        normalization="instancenorm",
    )
    model.train()
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 2, 16, 64, 64)


def test_vnet5_plus_deep_supervision_overlap_and_outer_convs():
    x = torch.randn(2, 1, 16, 64, 64)
    model = VNet(
        in_channels=1,
        num_classes=2,
        n_filters=16,
        num_stages=5,
        normalization="instancenorm",
        has_residual=False,
        has_dropout=False,
        deep_supervision=True,
        overlap_downsample=True,
        outer_convs=2,
    )
    # Verify overlapping downsample kernels/padding
    assert model.encoder.block_one_dw.conv[0].kernel_size == (1, 3, 3)
    assert model.encoder.block_one_dw.conv[0].padding == (0, 1, 1)
    assert model.encoder.block_three_dw.conv[0].kernel_size == (3, 3, 3)
    assert model.encoder.block_three_dw.conv[0].padding == (1, 1, 1)

    model.train()
    out = model(x)
    assert isinstance(out, dict)
    main_logits, aux_logits = extract_multiscale_outputs(out)
    assert main_logits.shape == (2, 2, 16, 64, 64)
    assert [tuple(t.shape) for t in aux_logits] == [
        (2, 2, 16, 32, 32),
        (2, 2, 16, 16, 16),
        (2, 2, 8, 8, 8),
    ]

    loss = main_logits.sum() + sum(a.sum() for a in aux_logits)
    loss.backward()

    model.eval()
    with torch.no_grad():
        eval_out = model(x)
    assert isinstance(eval_out, torch.Tensor)
    assert eval_out.shape == (2, 2, 16, 64, 64)


def test_vnet5_plus_model_builder_spec():
    cfg = {
        "experiment": {"num_classes": 2, "seed": 1337, "deterministic": True},
        "method": {"name": "urpc"},
        "model": {
            "name": "vnet",
            "in_channels": 1,
            "n_filters": 16,
            "num_stages": 5,
            "normalization": "instancenorm",
            "has_residual": False,
            "has_dropout": False,
            "deep_supervision": True,
            "overlap_downsample": True,
            "outer_convs": 2,
        },
        "data": {"patch_size": [32, 256, 256]},
    }
    spec = model_spec_from_config(cfg, num_classes=2)
    model = build_model_from_spec(spec, seed=1337)
    assert model.deep_supervision is True
    assert model.encoder.overlap_downsample is True
    assert model.encoder.outer_convs == 2
    assert model.decoder.outer_convs == 2
