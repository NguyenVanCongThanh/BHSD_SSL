"""Resolve configured global batches without modifying experiment configuration."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class BatchLayout:
    global_batch_size: int
    global_labeled_bs: int
    global_unlabeled_bs: int
    world_size: int
    local_batch_size: int
    local_labeled_bs: int
    local_unlabeled_bs: int

    def to_dict(self):
        return asdict(self)


def resolve_batch_layout(batch_size, labeled_bs, world_size=1, method='mean_teacher'):
    """Global sizes must divide evenly; CutMix methods require balanced pairs."""
    if world_size < 1 or batch_size < 1 or not 0 < labeled_bs <= batch_size:
        raise ValueError('Require world_size >= 1 and 0 < labeled_bs <= batch_size.')
    # Supervised uses complete batches drawn exclusively from the labeled pool.
    labeled = batch_size if method == 'supervised' else labeled_bs
    unlabeled = batch_size - labeled
    if labeled % world_size or unlabeled % world_size:
        raise ValueError(
            f'Global batch ({labeled} labeled + {unlabeled} unlabeled) must divide '
            f'evenly across world_size={world_size}; no rounding is performed.'
        )
    local_l, local_u = labeled // world_size, unlabeled // world_size
    if method != 'supervised' and (local_l < 1 or local_u < 1):
        raise ValueError(f'{method} requires labeled and unlabeled samples on every rank.')
    if method in ('pick', 'bcp') and (local_l < 2 or local_l % 2 or local_l != local_u):
        raise ValueError(
            f'{method} requires equal even labeled/unlabeled counts >= 2 per rank '
            f'for paired CutMix; got {local_l} labeled + {local_u} unlabeled. '
            f'For {world_size} ranks use global batch_size={4 * world_size}, '
            f'labeled_bs={2 * world_size} or another valid paired configuration.'
        )
    return BatchLayout(batch_size, labeled, unlabeled, world_size,
                       batch_size // world_size, local_l, local_u)
