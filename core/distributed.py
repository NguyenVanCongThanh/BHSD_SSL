"""
Distributed Data Parallel (DDP) utilities for MedSSL-BHSD.
Provides seamless environment detection, process group lifecycle management,
and communication primitives while leaving standard DP completely intact as default.
"""

from typing import Optional
import os
import logging
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    """Check if torch.distributed is available and current process group is initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """Get the number of processes in the distributed group, or 1 if not distributed."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get current process rank, or 0 if not distributed."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Check if current process is rank 0 (main process for logging & saving)."""
    return get_rank() == 0


def init_distributed_mode(ddp_requested: bool = False) -> bool:
    """
    Initialize distributed training mode if launched via torchrun or explicit --ddp flag.
    Returns:
        bool: True if distributed mode (DDP) was successfully initialized, False otherwise.
    """
    # Detect torchrun / torch.distributed launch environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif ddp_requested and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # Fallback or prompt if user passed --ddp without torchrun
        logging.warning("[DDP] --ddp flag passed but environment variables RANK/WORLD_SIZE not found. "
                        "Please launch via: torchrun --nproc_per_node=2 train.py --ddp ...")
        return False
    else:
        return False

    from datetime import timedelta
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist_backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    print(f"[DDP-Rank{rank}] Đang khởi tạo process group (local_rank={local_rank}, world_size={world_size}, backend={dist_backend})...", flush=True)
    logging.info(f"[DDP] Initializing process group: rank={rank}, world_size={world_size}, "
                 f"local_rank={local_rank}, backend={dist_backend}")
    
    # Anti-deadlock: Đặt timeout 1800s (30 phút) để Rank 1 có đủ thời gian chờ Rank 0 chạy sliding window validation (38 ca 3D)
    dist.init_process_group(
        backend=dist_backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=1800)
    )
    print(f"[DDP-Rank{rank}] Đang đồng bộ barrier khởi tạo...", flush=True)
    dist.barrier()
    print(f"[DDP-Rank{rank}] Khởi tạo DDP thành công! Đã vượt qua barrier.", flush=True)
    return True


def cleanup_distributed():
    """Destroy distributed process group on exit."""
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """Reduce a tensor across all distributed workers (All-Reduce)."""
    if not is_dist_avail_and_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= get_world_size()
    return rt
