from dataclasses import dataclass
import logging
from pathlib import Path

import modal
import click

from gpt2.train_gpt2 import GPT2TrainConfig, MODEL_CONFIGS, train_gpt2
from gpt2.model import GPT2Tokenizer
from gpt2.dataset import WikitextDataset

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


@app.function(
    image=image,
    gpu="A100",
    timeout=6 * 3600,
    volumes={VOL_MOUNT_PATH: volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_remote(config):
    train_gpt2(config)


@click.command()
@click.option(
    "--local", help="Whether training should run locally", type=bool, is_flag=True
)
@click.option(
    "--experiment-name", help="Unique name for an experiment", type=str, required=True
)
@click.option(
    "--model-config", help="Configuration of the GPT-2 model", type=click.Choice(MODEL_CONFIGS.keys()), required=True
)
@click.option(
    "--checkpoint-dir", help="Directory where checkpoints are stored", type=str
)
@click.option(
    "--checkpoint-freq",
    help="Save a checkpoint every n iterations",
    type=int,
)
@click.option(
    "--wandb-api-key",
    help="API key for Weights & Biases",
    type=str,
    envvar="WANDB_API_KEY",
)
def main(**kwargs):
    model_config = MODEL_CONFIGS[kwargs["model_config"]]
    config = GPT2TrainConfig(
        experiment_name=kwargs["experiment_name"],
        model_config=model_config,
        train_ds="wikitext-train",
        valid_ds="wikitext-valid",
        batch_size=16,
        ckpt_dir=Path(kwargs["checkpoint_dir"]) if kwargs["checkpoint_dir"] else None,
        ckpt_freq=kwargs["checkpoint_freq"],
    )

    if kwargs["local"]:
        train_gpt2(config)
    else:
        with app.run():
            train_remote.remote(config)


if __name__ == "__main__":
    main()
