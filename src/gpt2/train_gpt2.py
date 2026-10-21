#!/usr/bin/env python
# coding: utf-8
from dataclasses import dataclass
from pathlib import Path
import logging

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset, DataLoader

from gpt2.dataset import WikitextDataset
from gpt2.models.gpt2_model import GPT2Model, GPT2Tokenizer
from gpt2.train import AIAYNScheduler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPT2ModelConfig:
    d_model: int
    d_ff: int
    n_heads: int
    n_blocks: int
    max_context_length: int  # tok
    p_dropout: float = 0.1


# Local GPT-2 param config for training loop testing
MODEL_CONFIGS = {
    "gpt2-15m": GPT2ModelConfig(
        d_model=512,
        d_ff=1024,
        n_heads=4,
        n_blocks=4,
        max_context_length=256,
    ),
    "gpt2-base": GPT2ModelConfig(
        d_model=768,
        d_ff=3072,
        n_heads=12,
        n_blocks=12,
        max_context_length=512,
    ),
}

DATASETS = {
    "wikitext-train": WikitextDataset(split="train", tokenizer=GPT2Tokenizer()),
    "wikitext-valid": WikitextDataset(split="validation", tokenizer=GPT2Tokenizer()),
}

@dataclass(frozen=True)
class GPT2TrainConfig:
    experiment_name: str
    model_config: GPT2ModelConfig
    train_ds: Dataset
    batch_size: int
    valid_ds: Dataset | None = None
    ckpt_freq: int | None = None
    ckpt_dir: Path | None = None
    wandb_api_key: str | None = None

    def __post_init__(self) -> None:
        if self.ckpt_freq and not self.ckpt_dir:
            raise ValueError("ckpt_dir must be set if ckpt_freq is set")
        if not self.ckpt_freq and self.ckpt_dir:
            raise ValueError("ckpt_freq must be set if ckpt_dir is set")

        if self.ckpt_dir:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)


def format_ckpt_name(experiment_name: str, iteration: int) -> str:
    return f"ckpt_{experiment_name}_iter_{iteration}.pt"


class CheckpointExistsError(Exception):
    pass


def save_ckpt(
    path: Path,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    iteration: int,
    loss: int,
) -> Path:
    if path.exists():
        raise CheckpointExistsError(f"checkpoint {path} already exists")

    ckpt_dict = {
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "iteration": iteration,
        "train_loss": loss.item(),
    }

    with open(path, "wb") as f:
        torch.save(ckpt_dict, f)

    logger.info(f"saved checkpoint to {path}")
    return path


def train_gpt2(train_config: GPT2TrainConfig):
    wandb.init()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2Model(
        d_model=train_config.model_config.d_model,
        d_ff=train_config.model_config.d_ff,
        n_heads=train_config.model_config.n_heads,
        n_blocks=train_config.model_config.n_blocks,
        max_context_length=train_config.model_config.max_context_length,
        p_dropout=train_config.model_config.p_dropout,
    ).to(device)

    train_ds = DATASETS[train_config.train_ds]
    train_data_loader = DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        num_workers=2,
        pin_memory=True,
    )

    valid_data_loader = None
    valid_ds = None
    if train_config.valid_ds:
        valid_ds = DATASETS[train_config.valid_ds]
        valid_data_loader = (
            DataLoader(
                valid_ds,
                batch_size=train_config.batch_size,
                num_workers=2,
                pin_memory=True,
            )
    )

    optim = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)
    scheduler = AIAYNScheduler(optim, model.d_model)

    for iteration, batch in enumerate(train_data_loader):
        model.train()
        optim.zero_grad()

        tokens = batch.to(device, non_blocking=True)
        pad = torch.full((tokens.shape[0], 1), fill_value=-1).to(device)
        targets = torch.concat((tokens[:, 1:], pad), dim=1)
        logits = model.forward(tokens)

        # reshape the preds to a tensor of [all_tokens, vocab_size], which we'll then pair with the targets, which is [all_tokens]
        preds = logits.view(-1, model.tokenizer.vocab_size())
        targets = targets.view(-1)

        loss = F.cross_entropy(preds, targets, ignore_index=-1)
        loss.backward()
        optim.step()
        scheduler.step()

        if iteration % 50 == 0:
            logger.info(f"iter {iteration}: train loss: {loss.item()}")

            model.eval()
            with torch.no_grad():
                if valid_data_loader is not None:
                    valid_tokens = next(iter(valid_data_loader)).to(device)
                    valid_targets = torch.concat((valid_tokens[:, 1:], pad), dim=1)

                    valid_logits = model.forward(valid_tokens)

                    # reshape the preds to a tensor of [all_tokens, vocab_size], which we'll then pair with the targets, which is [all_tokens]
                    valid_preds = valid_logits.view(-1, model.tokenizer.vocab_size())
                    valid_targets = valid_targets.view(-1)
                    valid_loss = F.cross_entropy(
                        valid_preds, valid_targets, ignore_index=-1
                    )

                    logger.info(f"iter {iteration}: valid loss: {valid_loss.item()}")

                # Sample from the model for visual debugging
                temp = 0.7
                sampled = model.tokenizer.encode("<en>").unsqueeze(0).to(device)  # d
                while sampled.shape[1] < model.max_context_length:
                    logits = (
                        model.forward(sampled) / temp
                    )  # [batch, seq_len, vocab_size]

                    # softmax along vocab_size dimension, then select out distribution for last token
                    preds = logits.softmax(dim=-1)[:, -1, :]
                    next_tok = torch.multinomial(preds, num_samples=1)

                    # add to autoregressive sequence
                    sampled = torch.cat((sampled, next_tok), dim=1)

                    print(model.tokenizer.decode(next_tok.squeeze(0)), end="")

                    if next_tok.item() == model.tokenizer.end_of_text_token():
                        print("")
                        break
                print("")

        if train_config.ckpt_freq and iteration % train_config.ckpt_freq == 0:
            ckpt_path = train_config.ckpt_dir / format_ckpt_name(
                train_config.experiment_name, iteration
            )
            save_ckpt(ckpt_path, model, optim, scheduler, iteration, loss.item())
