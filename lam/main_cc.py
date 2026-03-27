import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import torch
# Allow all globals when loading checkpoints (needed for functools.partial,
# jsonargparse internals, etc. that Lightning embeds in .ckpt files)
torch.serialization.add_safe_globals([object])
# Monkey-patch to disable weights_only restriction on checkpoint loading
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load
# Disable WandB on non-rank-0 processes to prevent all ranks from initializing
# the same run ID simultaneously, which causes a 30-min DDP barrier hang.
if os.environ.get("LOCAL_RANK", "0") != "0":
    os.environ["WANDB_MODE"] = "disabled"

from lightning.pytorch.cli import LightningCLI

from lam.dataset_cc import LightningVideoDataset_cc
from lam.model import LAM

cli = LightningCLI(
    LAM,
    LightningVideoDataset_cc,
    seed_everything_default=32
)
