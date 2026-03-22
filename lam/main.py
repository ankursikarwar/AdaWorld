import os
# Disable WandB on non-rank-0 processes to prevent all ranks from initializing
# the same run ID simultaneously, which causes a 30-min DDP barrier hang.
if os.environ.get("LOCAL_RANK", "0") != "0":
    os.environ["WANDB_MODE"] = "disabled"

from lightning.pytorch.cli import LightningCLI

from lam.dataset import LightningVideoDataset
from lam.model import LAM

cli = LightningCLI(
    LAM,
    LightningVideoDataset,
    seed_everything_default=32
)
