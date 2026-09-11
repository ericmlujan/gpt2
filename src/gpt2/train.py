import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import modal
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import wandb

from gpt2.models.translation_model import TransformerTranslationModel
from gpt2.model import GPT2Tokenizer

logger = logging.getLogger("train")


@dataclass
class TrainConfig:
    bs: int
    epochs: int
    experiment_name: str
    checkpoint_dir: Optional[Path] = None


class TranslationDataset(Dataset):
    def __init__(self, split: str) -> None:
        self._ds = load_dataset("Helsinki-NLP/opus-100", "en-es", split=split)
        print(f"split: {split}, {len(self._ds)}")

    def __getitem__(self, index: int) -> tuple[str, str]:
        return (
            self._ds[index]["translation"]["en"],
            self._ds[index]["translation"]["es"],
        )

    def __len__(self) -> int:
        return len(self._ds)


MAX_SEQ_LEN = 254  # 256 if you account for the <EOT> tokens wrapping the inner string
VALID_LOSS_STEPS = 1000  # How many steps to evaluate and log valid loss to wandb


class AIAYNScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Learning rate scheduler that implements the linear warmup -> inverse square root
    strategy from _Attention is All You Need_.
    """

    def __init__(self, optim: torch.optim.Adam, d_model: int, warmup_steps: int = 4000):
        self.optim = optim
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optim)

    def get_lr(self) -> list[float | torch.Tensor]:
        num_steps = self.last_epoch + 1
        lr = (
            1.0
            / math.pow(self.d_model, 0.5)
            * min(
                math.pow(num_steps, -0.5), num_steps * math.pow(self.warmup_steps, -1.5)
            )
        )
        return [lr]


def valid_loss(
    model: TransformerTranslationModel, valid_dl: DataLoader, device
) -> float:
    model.eval()
    total_loss = 0
    num_samples = 0

    for batch in valid_dl:
        if batch is None:
            continue

        encoder_inputs, decoder_inputs, targets = (t.to(device) for t in batch)
        with torch.no_grad():
            logits = model.forward(encoder_inputs, decoder_inputs)
            loss = F.cross_entropy(
                logits.view(-1, model.tokenizer.vocab_size()),
                targets.view(-1),
                ignore_index=model.tokenizer.pad_token(),
            )
            total_loss += loss.item() * encoder_inputs.shape[0]
            num_samples += encoder_inputs.shape[0]

    model.train()
    return total_loss / num_samples if total_loss else 0.0


def collate(
    tokenizer: GPT2Tokenizer, batch: list[tuple[str, str]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    encoder_inputs, decoder_inputs, targets = [], [], []
    pad = tokenizer.pad_token()

    for en, es in batch:
        en_tok = tokenizer.encode(en)
        es_tok = tokenizer.encode(es)
        if len(en_tok) > MAX_SEQ_LEN or len(es_tok) > MAX_SEQ_LEN:
            continue

        # English source: <EOT>Hello world<EOT>
        encoder_inputs.append(tokenizer.wrap_eot(en_tok))
        # Spanish target shifted by one: decoder sees <EOT>+seq, predicts seq+<EOT>
        dec = tokenizer.prepend_eot(es_tok)
        decoder_inputs.append(dec)
        targets.append(tokenizer.shift_right(dec))

    if not encoder_inputs:
        return None

    return (
        pad_sequence(encoder_inputs, batch_first=True, padding_value=pad),
        pad_sequence(decoder_inputs, batch_first=True, padding_value=pad),
        pad_sequence(targets, batch_first=True, padding_value=pad),
    )


def train(config: TrainConfig, volume: Optional[modal.Volume] = None):
    wandb.init(
        project="gpt2",
        config={"bs": config.bs, "epochs": config.epochs},
        name=config.experiment_name,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerTranslationModel().to(device)

    # Hyperparameters taken from _Attention is All You Need_ paper
    optim = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)
    scheduler = AIAYNScheduler(optim, model.D_MODEL)

    start_epoch = 0

    if config.checkpoint_dir:
        # Ensure that the output directory exists
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = config.checkpoint_dir / "latest.tar"
        if not checkpoint_path.exists():
            logger.warning(
                "Checkpoint path was specified but no latest checkpoint exists; skipping load and training from scratch"
            )
        else:
            logger.info(f"Restoring checkpoint from {checkpoint_path}")
            with open(checkpoint_path, "rb") as f:
                ckpt = torch.load(f, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                optim.load_state_dict(ckpt["optim_state_dict"])
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = ckpt["epoch"]

    translation_dataset = TranslationDataset(split="train")
    translation_dl = DataLoader(
        translation_dataset,
        batch_size=config.bs,
        collate_fn=lambda batch: collate(model.tokenizer, batch),
    )

    valid_dataset = TranslationDataset(split="test")
    valid_dl = DataLoader(
        valid_dataset,
        batch_size=config.bs,
        collate_fn=lambda batch: collate(model.tokenizer, batch),
    )

    logger.info("Starting training")

    model.train()
    for epoch in range(start_epoch, config.epochs):
        for i, batch in enumerate(translation_dl):
            if batch is None:  # whole batch was filtered out
                continue

            encoder_inputs, decoder_inputs, targets = (t.to(device) for t in batch)

            optim.zero_grad()

            logits = model.forward(encoder_inputs, decoder_inputs)

            # tensor.view() will flatten the logits from [batch, seq_len, vocab_size] to [batch * seq_len, vocab_size].
            # It will flatten the targets from [batch, seq_len, 1] to [batch * seq_len], where each value is a target token index
            loss = F.cross_entropy(
                logits.view(-1, model.tokenizer.vocab_size()),
                targets.view(-1),
                ignore_index=model.tokenizer.pad_token(),
                label_smoothing=0.1,
            )
            loss.backward()

            # Prevent gradients from getting too too crazy
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optim.step()
            scheduler.step()
            wandb.log({"train/loss": loss.item()})

            if i != 0 and i % VALID_LOSS_STEPS == 0:
                vl = valid_loss(model, valid_dl, device)
                logger.info(f"valid loss: {vl}")
                wandb.log({"val/loss": vl})

        # End of epoch, log valid loss and save checkpoint
        epoch_loss = valid_loss(model, valid_dl, device)
        wandb.log({"val/loss_epoch": epoch_loss, "epoch": epoch})

        if config.checkpoint_dir:
            train_state_dict = {
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1,
            }
            epoch_file = f"{epoch}.tar"
            output_path = config.checkpoint_dir / epoch_file

            logger.info(f"Epoch {epoch} completed, saving checkpoint to {output_path}")

            with open(output_path, "wb") as f:
                torch.save(train_state_dict, f)

            logger.info(f"Checkpoint saved to {output_path}")

            # Symlink latest.tar for resume
            latest_path = config.checkpoint_dir / "latest.tar"
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(epoch_file)

            # Sync volume with Modal's volume storage
            if volume:
                volume.commit()
