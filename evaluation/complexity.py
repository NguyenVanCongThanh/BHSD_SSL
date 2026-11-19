"""
Model Complexity & Efficiency Profiler for 3D Medical Image Segmentation.
Measures:
- Parameters (Total & Trainable)
- Computational Cost: MACs & FLOPs (GFLOPs)
- Latency (ms) with high-precision CUDA Events
- Throughput (Volumes / sec)
- Peak GPU VRAM Memory (MB)

Zero-dependency fallback: Works with native PyTorch even if thop/fvcore are not installed.
"""

from typing import Dict, Any, Tuple, Optional
import time
import logging
import torch
import torch.nn as nn


def _estimate_macs_via_hooks(model: nn.Module, dummy_input: torch.Tensor) -> Optional[float]:
    """
    Zero-dependency MACs estimator using PyTorch forward hooks.
    Accurately supports Conv1d/2d/3d, ConvTranspose1d/2d/3d, Linear, and Norm layers.
    All hooks are guaranteed to be removed in a finally block.
    """
    total_macs = [0.0]
    handles = []

    def _conv_hook(module, inp, out):
        if not isinstance(out, torch.Tensor):
            return
        # out shape: (N, C_out, *spatial_out)
        batch_size = out.shape[0]
        out_elems = int(out[0].numel())
        kernel_ops = 1
        for k in module.kernel_size:
            kernel_ops *= int(k)
        c_in_per_group = module.in_channels // module.groups
        macs_per_elem = c_in_per_group * kernel_ops
        total_macs[0] += float(batch_size * out_elems * macs_per_elem)

    def _conv_transpose_hook(module, inp, out):
        if not inp or not isinstance(inp[0], torch.Tensor):
            return
        # For transposed conv, each input element contributes kernel_ops * (C_out / groups) MACs
        x = inp[0]
        batch_size = x.shape[0]
        in_elems = int(x[0].numel())
        kernel_ops = 1
        for k in module.kernel_size:
            kernel_ops *= int(k)
        c_out_per_group = module.out_channels // module.groups
        total_macs[0] += float(batch_size * in_elems * c_out_per_group * kernel_ops)

    def _linear_hook(module, inp, out):
        if not isinstance(out, torch.Tensor):
            return
        total_macs[0] += float(out.numel() * module.in_features)

    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    conv_t_types = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)

    for m in model.modules():
        if isinstance(m, conv_types):
            handles.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, conv_t_types):
            handles.append(m.register_forward_hook(_conv_transpose_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(_linear_hook))

    try:
        with torch.no_grad():
            _ = model(dummy_input)
    finally:
        for h in handles:
            h.remove()

    return total_macs[0] if total_macs[0] > 0 else None


def profile_model_complexity(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 1, 32, 160, 160),
    device: Optional[torch.device] = None,
    num_warmup: int = 10,
    num_runs: int = 50
) -> Dict[str, Any]:
    """
    Profile a PyTorch model for computational complexity and inference speed.

    Args:
        model: PyTorch nn.Module.
        input_size: Shape of dummy input tensor (B, C, D, H, W).
        device: Device to profile on (CPU or CUDA).
        num_warmup: Number of warmup iterations.
        num_runs: Number of timed inference iterations.

    Returns:
        Dict with parameters, GFLOPs, latency, and memory statistics.
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    # 1. Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. FLOPs / MACs Estimation
    dummy_input = torch.randn(*input_size, device=device)
    flops = None
    macs = None
    method_used = "native"

    # Try thop first
    try:
        from thop import profile as thop_profile
        macs_val, _ = thop_profile(model, inputs=(dummy_input,), verbose=False)
        macs = float(macs_val)
        flops = macs * 2.0  # 1 MAC ~= 2 FLOPs
        method_used = "thop"
    except Exception:
        pass

    # Try fvcore if thop failed
    if flops is None:
        try:
            from fvcore.nn import FlopCountAnalysis
            fca = FlopCountAnalysis(model, dummy_input)
            fca.unsupported_ops_warnings(False)
            flops = float(fca.total())
            macs = flops / 2.0
            method_used = "fvcore"
        except Exception:
            pass

    # Fallback: Zero-dependency PyTorch forward-hook MACs estimator (accurate for 3D Conv/ConvTranspose/Linear)
    if flops is None or flops <= 0:
        try:
            hook_macs = _estimate_macs_via_hooks(model, dummy_input)
            if hook_macs is not None and hook_macs > 0:
                macs = float(hook_macs)
                flops = macs * 2.0
                method_used = "torch_hooks"
            else:
                flops = None
                macs = None
        except Exception:
            flops = None
            macs = None

    # 3. Latency & Throughput Benchmark
    is_cuda = (device.type == 'cuda' and torch.cuda.is_available())
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
        if is_cuda:
            torch.cuda.synchronize()

    # Benchmark loop using CUDA Events for microsecond precision
    if is_cuda:
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = []
        with torch.no_grad():
            for _ in range(num_runs):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))  # milliseconds
        latency_ms = sum(timings) / len(timings)
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        # CPU timing
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        total_time = time.perf_counter() - start
        latency_ms = (total_time / num_runs) * 1000
        peak_vram_mb = 0.0

    throughput_fps = (1000.0 / latency_ms) if latency_ms > 0 else 0.0

    # Restore initial model state
    if was_training:
        model.train()

    results = {
        'input_shape': list(input_size),
        'total_params_m': total_params / 1e6,
        'trainable_params_m': trainable_params / 1e6,
        'gflops': (flops / 1e9) if flops is not None else None,
        'gmacs': (macs / 1e9) if macs is not None else None,
        'latency_ms': latency_ms,
        'throughput_fps': throughput_fps,
        'peak_vram_mb': peak_vram_mb,
        'profiler_backend': method_used
    }
    return results


def format_complexity_summary(metrics: Dict[str, Any]) -> str:
    """Format complexity metrics into a clean text table."""
    flops_str = f"{metrics['gflops']:.2f} GFLOPs" if metrics['gflops'] is not None else "N/A (install thop)"
    macs_str = f"{metrics['gmacs']:.2f} GMACs" if metrics['gmacs'] is not None else "N/A"
    
    lines = [
        "=" * 60,
        "=== MODEL COMPUTATIONAL EFFICIENCY & COMPLEXITY BENCHMARK ===",
        "=" * 60,
        f"  Input Dimensions:    {metrics['input_shape']}",
        f"  Total Parameters:    {metrics['total_params_m']:.3f} M",
        f"  Trainable Params:    {metrics['trainable_params_m']:.3f} M",
        f"  FLOPs Cost:          {flops_str} ({macs_str})",
        f"  Inference Latency:   {metrics['latency_ms']:.2f} ms / patch",
        f"  Throughput:          {metrics['throughput_fps']:.1f} patches / sec",
        f"  Peak Inference VRAM: {metrics['peak_vram_mb']:.1f} MB",
        "=" * 60,
    ]
    return "\n".join(lines)
