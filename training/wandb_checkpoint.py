import os

import wandb

from utils.config import cfg, resolve

wandb.login()

api = wandb.Api()
artifact = api.artifact(cfg["wandb_artifact"])
artifact.download(root=resolve(os.path.dirname(cfg["checkpoint"])))