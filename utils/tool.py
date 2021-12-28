import wandb
from wandb.wandb_run import Run


def get_wandb(configuration: dict) -> Run:
    run = wandb.init(**configuration)
    if isinstance(run, Run):
        return run
    else:
        raise Exception
