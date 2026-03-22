import os

# Disable WandB on non-rank-0 processes to prevent DDP barrier hang
if os.environ.get("LOCAL_RANK", "0") != "0":
    os.environ["WANDB_MODE"] = "disabled"

from lightning.pytorch.cli import LightningCLI
from lam.dataset_ssv2_clip import LightningSSv2ClipDataset
from lam.model_clip_align import LAMClipAlign

cli = LightningCLI(LAMClipAlign, LightningSSv2ClipDataset, seed_everything_default=32)
