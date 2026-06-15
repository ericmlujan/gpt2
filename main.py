import modal
import click

from gpt2.train import train

# app = modal.App.lookup("gpt2", create_if_missing=True)
app = modal.App("gpt2")
image = (
    modal.Image.from_registry("pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel")
    .uv_pip_install("datasets", "tqdm", "tiktoken")
    .add_local_python_source("gpt2")
)


@app.function(image=image, gpu="T4")
def train_modal():
    train()


@click.command()
@click.option(
    "--local", help="Whether training should run locally", type=bool, is_flag=True
)
def main(**kwargs):
    if kwargs["local"]:
        train()
    else:
        with app.run():
            train_modal.remote()


if __name__ == "__main__":
    main()
