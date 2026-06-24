import modal
import click

from gpt2.train import TrainConfig, train

# app = modal.App.lookup("gpt2", create_if_missing=True)
app = modal.App("gpt2")
image = (
    modal.Image.from_registry("pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel")
    .uv_pip_install("datasets", "tqdm", "tiktoken")
    .add_local_python_source("gpt2")
)


modal_train_config = TrainConfig(bs=56, epochs=10)
local_train_config = TrainConfig(bs=16, epochs=2)


@app.function(image=image, gpu="A100", timeout=3600)
def train_remote():
    train(modal_train_config)


@click.command()
@click.option(
    "--local", help="Whether training should run locally", type=bool, is_flag=True
)
def main(**kwargs):
    if kwargs["local"]:
        train(local_train_config)
    else:
        with modal.enable_output():
            with app.run():
                train_remote.remote()


if __name__ == "__main__":
    main()
