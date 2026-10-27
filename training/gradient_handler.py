"""
Coordinated Gradient and AMP Handler for Multi-Model, Multi-Optimizer Volumetric Training.
Ensures DDP consensus, overflow protection, and coordinated multi-optimizer updates.
"""

from typing import List, Callable, Optional, Any
import logging
import torch
import torch.nn as nn
from utils.checkpoint import get_raw_model


class GradientHandler:
    """
    Coordinates gradient backpropagation, all-rank consensus, clipping,
    and parameter updates across arbitrary sets of models and optimizers.
    """
    def __init__(
        self,
        optimizers: List[torch.optim.Optimizer],
        models: List[nn.Module],
        scaler: Optional[Any] = None,
        use_amp: bool = False,
        is_ddp: bool = False,
        device: torch.device = torch.device('cpu'),
        clip_norm: float = 12.0,
        after_step_fn: Optional[Callable[[int], None]] = None
    ):
        self.optimizers = optimizers
        self.models = models
        self.scaler = scaler
        self.use_amp = use_amp
        self.is_ddp = is_ddp
        self.device = device
        self.clip_norm = clip_norm
        self.after_step_fn = after_step_fn
        self.successful_updates = 0

    def apply_gradients(self, loss: torch.Tensor, iter_num: int) -> bool:
        if loss.ndim != 0:
            raise ValueError('Training loss must be a scalar')
        loss_finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
        if self.is_ddp:
            torch.distributed.all_reduce(loss_finite, op=torch.distributed.ReduceOp.MIN)
        if not loss_finite.item():
            raise FloatingPointError(f'Invalid training loss at iteration {iter_num}')

        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            for optimizer in self.optimizers:
                self.scaler.unscale_(optimizer)

            is_finite = torch.ones((), device=self.device, dtype=torch.int32)
            for optimizer in self.optimizers:
                for group in optimizer.param_groups:
                    for param in group['params']:
                        if param.grad is not None:
                            is_finite.mul_(torch.isfinite(param.grad).all().to(torch.int32))
            if self.is_ddp:
                torch.distributed.all_reduce(is_finite, op=torch.distributed.ReduceOp.MIN)

            scale_before = self.scaler.get_scale()
            if is_finite.item() == 0:
                new_scale = scale_before * self.scaler.get_backoff_factor()
                self.scaler.update(new_scale=new_scale)
                logging.warning(
                    '[AMP] Iteration %d: Nonfinite gradient overflow detected across optimizers/ranks; '
                    'all optimizer steps skipped, scale adjusted from %.1f to %.1f',
                    iter_num, scale_before, new_scale
                )
                if new_scale < 1e-4:
                    raise FloatingPointError(f'GradScaler scale underflow at iteration {iter_num}: model or loss diverged')
                return False
            else:
                for m in self.models:
                    raw_m = get_raw_model(m)
                    torch.nn.utils.clip_grad_norm_(raw_m.parameters(), max_norm=self.clip_norm)

                for optimizer in self.optimizers:
                    self.scaler.step(optimizer)
                self.scaler.update()
                if self.after_step_fn:
                    self.after_step_fn(iter_num)
                self.successful_updates += 1
                return True
        else:
            loss.backward()
            for m in self.models:
                raw_m = get_raw_model(m)
                torch.nn.utils.clip_grad_norm_(raw_m.parameters(), max_norm=self.clip_norm)

            finite = torch.ones((), device=self.device, dtype=torch.int32)
            for optimizer in self.optimizers:
                for group in optimizer.param_groups:
                    for param in group['params']:
                        if param.grad is not None:
                            finite.mul_(torch.isfinite(param.grad).all().to(torch.int32))
            if self.is_ddp:
                torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(f'Nonfinite gradients at iteration {iter_num}')

            for optimizer in self.optimizers:
                optimizer.step()
            if self.after_step_fn:
                self.after_step_fn(iter_num)
            self.successful_updates += 1
            return True

