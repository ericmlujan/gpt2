import modal
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from gpt2.model import GPT2Model, GPT2Tokenizer

app = modal.App("gpt2")
image = (
    modal.Image.from_registry("pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel")
    .uv_pip_install("datasets", "tqdm", "tiktoken")
    .add_local_python_source("gpt2")
)


class TranslationDataset(Dataset):
    def __init__(self, split: str) -> None:
        self._ds = load_dataset("Helsinki-NLP/opus-100", "en-es")[split]

    def __getitem__(self, index: int) -> tuple[str, str]:
        return (
            self._ds[index]["translation"]["en"],
            self._ds[index]["translation"]["es"],
        )

    def __len__(self) -> int:
        return len(self._ds)


def shift_right(seq: torch.Tensor, eot_tok: int) -> torch.Tensor:
    # Lop off the first token, append the EOT token
    return torch.concat([seq[1:], torch.tensor(eot_tok).unsqueeze(dim=0)])


def prepend_eot(seq: torch.Tensor, eot_tok: int) -> torch.Tensor:
    return torch.concat([torch.tensor(eot_tok).unsqueeze(dim=0), seq])


def wrap_eot(seq: torch.Tensor, eot_tok: int) -> torch.Tensor:
    eot_tensor = torch.tensor(eot_tok).unsqueeze(dim=0)
    return torch.concat([eot_tensor, seq, eot_tensor])


MAX_SEQ_LEN = 254


def make_collate_fn(tokenizer: GPT2Tokenizer):
    eot = tokenizer.end_of_text_token()
    pad = tokenizer.pad_token()

    def collate(batch: list[tuple[str, str]]):
        encoder_inputs, decoder_inputs, targets = [], [], []
        for en, es in batch:
            en_tok = tokenizer.encode(en)
            es_tok = tokenizer.encode(es)
            if len(en_tok) > MAX_SEQ_LEN or len(es_tok) > MAX_SEQ_LEN:
                continue

            # English source: <EOT>Hello world<EOT>
            encoder_inputs.append(wrap_eot(en_tok, eot))
            # Spanish target shifted by one: decoder sees <EOT>+seq, predicts seq+<EOT>
            dec = prepend_eot(es_tok, eot)
            decoder_inputs.append(dec)
            targets.append(shift_right(dec, eot))

        if not encoder_inputs:
            return None

        return (
            pad_sequence(encoder_inputs, batch_first=True, padding_value=pad),
            pad_sequence(decoder_inputs, batch_first=True, padding_value=pad),
            pad_sequence(targets, batch_first=True, padding_value=pad),
        )

    return collate


@app.function(image=image, gpu="T4")
def train():
    TRAIN_BS = 24
    TRAIN_EPOCHS = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2Model().to(device)

    translation_dataset = TranslationDataset(split="train")
    translation_dl = DataLoader(
        translation_dataset,
        batch_size=TRAIN_BS,
        collate_fn=make_collate_fn(model.tokenizer),
    )

    optim = torch.optim.AdamW(model.parameters())

    for _ in trange(TRAIN_EPOCHS):
        for i, batch in enumerate(tqdm(translation_dl)):
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
            )
            loss.backward()
            optim.step()

            if i % 1000 == 0:
                print(f"train loss: {loss}")
