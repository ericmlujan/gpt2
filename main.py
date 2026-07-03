import dataclasses
import logging
from pathlib import Path

import modal
import click

from gpt2.train import TrainConfig, train

logging.basicConfig(level=logging.INFO)

# app = modal.App.lookup("gpt2", create_if_missing=True)
app = modal.App("gpt2")
image = (
    modal.Image.from_registry("pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel")
    .uv_pip_install("datasets", "tqdm", "tiktoken", "wandb")
    .add_local_python_source("gpt2")
)

volume = modal.Volume.from_name("gpt2")
VOL_MOUNT_PATH = Path("/vol")

modal_train_config = TrainConfig(bs=56, epochs=6, experiment_name="")
local_train_config = TrainConfig(bs=16, epochs=2, experiment_name="")


@app.function(
    image=image,
    gpu="A100",
    timeout=6 * 3600,
    volumes={VOL_MOUNT_PATH: volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_remote(config: TrainConfig):
    train(config, volume)


@click.command()
@click.option(
    "--local", help="Whether training should run locally", type=bool, is_flag=True
)
@click.option(
    "--experiment-name", help="Unique name for an experiment", type=str, required=True
)
@click.option(
    "--checkpoint-dir", help="Directory where checkpoints are stored", type=str
)
def main(**kwargs):

    if kwargs["local"]:
        checkpoint_dir = (
            Path(kwargs["checkpoint_dir"]) / Path(kwargs["experiment_name"])
            if kwargs["checkpoint_dir"]
            else None
        )
        config = dataclasses.replace(
            local_train_config,
            checkpoint_dir=checkpoint_dir,
            experiment_name=kwargs["experiment_name"],
        )
        train(config)
    else:
        with app.run(detach=True):
            checkpoint_dir = (
                (
                    VOL_MOUNT_PATH
                    / Path(kwargs["checkpoint_dir"])
                    / Path(kwargs["experiment_name"])
                )
                if kwargs["checkpoint_dir"]
                else None
            )
            config = dataclasses.replace(
                modal_train_config,
                checkpoint_dir=checkpoint_dir,
                experiment_name=kwargs["experiment_name"],
            )
            train_remote.spawn(config)


if __name__ == "__main__":
    main()
