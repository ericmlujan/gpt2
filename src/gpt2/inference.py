from pathlib import Path

import torch
import click

from gpt2.model import TransformerTranslationModel

@click.option("--checkpoint", help="Path to the checkpoint to load", type=Path, required=True)
@click.command()
def main(**kwargs):
    ckpt_path = kwargs["checkpoint"]
    if not ckpt_path.exists():
        raise RuntimeError("checkpoint path don't exist")

    with open(ckpt_path, 'rb') as f:
        map_location = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ckpt = torch.load(f, weights_only=False, map_location=map_location)

    print(f"Loaded checkpoint from epoch {ckpt["epoch"]}")
    model = TransformerTranslationModel()
    model.load_state_dict(ckpt["model_state_dict"])

    # Set model to eval mode, to deactivate dropout and all that
    model.eval()

    print("Bienvendios al traductor Lujan, buena suerte...")
    while(True):
        translation_request = input("> ").strip()
        translation = model.translate(translation_request)
        print(translation)


if __name__ == "__main__":
    main()
