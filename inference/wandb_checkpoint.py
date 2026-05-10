import wandb

wandb.login()

api = wandb.Api()
artifact = api.artifact("dlaoszu/si-tone/conformer-checkpoint:latest")
artifact.download(root="./checkpoints")