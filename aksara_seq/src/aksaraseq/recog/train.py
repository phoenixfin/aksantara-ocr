"""Training and evaluation for the CRNN+CTC baseline."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import LineDataset, WidthBucketSampler, collate
from .metrics import ErrorRates, factor_errors, greedy_decode
from .model import build_model, count_parameters
from .vocab import Vocab, WORD_SEP


@dataclass
class TrainConfig:
    corpus: Path = Path("aksara_seq/build/corpus/v1")
    script: str = "Sunda"
    head: str = "plain"                  # plain | factored
    out: Path = Path("aksara_seq/build/recog")

    height: int = 64
    batch_size: int = 32
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.03
    grad_clip: float = 5.0
    dropout: float = 0.1
    rnn_hidden: int = 256
    rnn_layers: int = 2
    conv_dim: int = 512

    augment: bool = True
    num_workers: int = 2
    seed: int = 0
    amp: bool = True
    limit_train: int = 0                 # 0 = all; for smoke tests
    limit_eval: int = 0
    holdout_cells: list = field(default_factory=list)   # syllables to withhold
    patience: int = 8
    # Directory name for this run. Left empty it is "<script>_<head>", which
    # collides across seeds when several run under one output root.
    tag: str = ""
    train_frac: float = 1.0            # recorded for the data-scaling curve

    def as_dict(self) -> dict:
        d = asdict(self)
        d["corpus"], d["out"] = str(self.corpus), str(self.out)
        return d


def _device(cfg: TrainConfig) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(cfg: TrainConfig, vocab: Vocab):
    holdout = set(cfg.holdout_cells)
    train = LineDataset(cfg.corpus, cfg.script, "train", vocab,
                        height=cfg.height, holdout=holdout,
                        augment=cfg.augment, limit=cfg.limit_train)
    val = LineDataset(cfg.corpus, cfg.script, "val", vocab,
                      height=cfg.height, limit=cfg.limit_eval)
    test = LineDataset(cfg.corpus, cfg.script, "test", vocab,
                       height=cfg.height, limit=cfg.limit_eval)

    def loader(ds, shuffle):
        sampler = WidthBucketSampler(ds.widths, cfg.batch_size,
                                     shuffle=shuffle, seed=cfg.seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                          num_workers=cfg.num_workers,
                          pin_memory=torch.cuda.is_available()), sampler

    (train_dl, train_sampler) = loader(train, True)
    (val_dl, _) = loader(val, False)
    (test_dl, _) = loader(test, False)
    return (train, val, test), (train_dl, val_dl, test_dl), train_sampler


@torch.no_grad()
def evaluate(model, loader, vocab, device, amp: bool = False,
             holdout=None) -> dict:
    """Evaluate, optionally scoring held-out syllables separately.

    Held-out cells are scored by *multiset recall* -- of every occurrence of a
    withheld syllable in the reference, how many does the prediction contain
    -- rather than by position-wise accuracy. Position-wise scoring needs the
    two sequences to be the same length, and a model that cannot emit a
    syllable at all produces deletions, so exactly the lines that matter would
    be the ones excluded. Recall is alignment-free and answers the actual
    question: can this model produce the syllable it never saw?
    """
    model.eval()
    rates = ErrorRates()
    onset_err = vowel_err = factor_n = 0
    holdout = set(holdout or ())
    held_hit = held_ref = seen_hit = seen_ref = 0

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        widths = batch["widths"].to(device)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(images)
        input_lengths = model.input_lengths(widths).cpu()
        input_lengths = torch.clamp(input_lengths, max=logits.shape[1])

        preds = greedy_decode(logits.float().cpu(), input_lengths)
        offset = 0
        for i, seq in enumerate(preds):
            n = int(batch["target_lengths"][i])
            ref_idx = batch["targets"][offset: offset + n].tolist()
            offset += n
            pred_tokens = [vocab.token(k) for k in seq]
            ref_tokens = [vocab.token(k) for k in ref_idx]
            rates.update(pred_tokens, ref_tokens, tag=batch["styles"][i])
            oe, ve, cn = factor_errors(pred_tokens, ref_tokens, vocab)
            onset_err += oe
            vowel_err += ve
            factor_n += cn

            if holdout:
                pred_count = Counter(pred_tokens)
                for tok, n_ref in Counter(ref_tokens).items():
                    if tok == WORD_SEP:
                        continue
                    hit = min(pred_count.get(tok, 0), n_ref)
                    if tok in holdout:
                        held_ref += n_ref
                        held_hit += hit
                    else:
                        seen_ref += n_ref
                        seen_hit += hit

    out = rates.summary()
    if factor_n:
        out["onset_error"] = onset_err / factor_n
        out["vowel_error"] = vowel_err / factor_n
        out["factor_aligned_tokens"] = factor_n
    if holdout:
        out["holdout_recall"] = held_hit / max(held_ref, 1)
        out["seen_recall"] = seen_hit / max(seen_ref, 1)
        out["holdout_tokens"] = held_ref
        out["seen_tokens"] = seen_ref
    return out


def train(cfg: TrainConfig, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    device = _device(cfg)

    charset = Path(cfg.corpus) / cfg.script / "charset.json"
    vocab = Vocab.from_charset(charset)
    log(f"{cfg.script}: {vocab.summary()}")
    if cfg.holdout_cells:
        log(f"withholding {len(cfg.holdout_cells)} syllables from training: "
            f"{sorted(cfg.holdout_cells)[:12]}"
            + (" ..." if len(cfg.holdout_cells) > 12 else ""))

    (train_ds, val_ds, test_ds), (train_dl, val_dl, test_dl), sampler = \
        make_loaders(cfg, vocab)
    log(f"lines  train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")

    model = build_model(vocab, head=cfg.head, conv_dim=cfg.conv_dim,
                        rnn_hidden=cfg.rnn_hidden, rnn_layers=cfg.rnn_layers,
                        dropout=cfg.dropout).to(device)
    log(f"head={cfg.head}  parameters={count_parameters(model):,}  device={device}")

    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, len(sampler))
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total_steps * cfg.warmup_frac))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(cfg.out) / (cfg.tag or f"{cfg.script}_{cfg.head}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(cfg.as_dict(), indent=2), encoding="utf-8")

    history, best, best_epoch, step = [], math.inf, -1, 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        sampler.set_epoch(epoch)
        running, seen = 0.0, 0

        for batch in train_dl:
            images = batch["images"].to(device, non_blocking=True)
            widths = batch["widths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            with torch.autocast("cuda", enabled=use_amp):
                logits = model(images)
            input_lengths = torch.clamp(model.input_lengths(widths),
                                        max=logits.shape[1])
            # CTC needs (T,B,C) log-probs, and float32 for numerical stability
            log_probs = logits.float().log_softmax(-1).permute(1, 0, 2)
            loss = ctc(log_probs, targets, input_lengths.cpu(),
                       target_lengths.cpu())

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1

            running += loss.detach().item() * images.size(0)
            seen += images.size(0)

        val = evaluate(model, val_dl, vocab, device, amp=use_amp)
        row = {"epoch": epoch, "loss": running / max(seen, 1),
               "lr": sched.get_last_lr()[0], **{k: v for k, v in val.items()
                                                if k != "by_style"}}
        history.append(row)
        log(f"epoch {epoch:3d}  loss {row['loss']:.4f}  "
            f"val SER {val['ser']:.4f}  WER {val['wer']:.4f}  "
            f"line-acc {val['line_accuracy']:.3f}  "
            f"({time.time() - t0:.0f}s)")

        if val["ser"] < best:
            best, best_epoch = val["ser"], epoch
            torch.save({"model": model.state_dict(), "config": cfg.as_dict(),
                        "epoch": epoch, "val": val}, out_dir / "best.pt")
        elif epoch - best_epoch >= cfg.patience:
            log(f"no val improvement for {cfg.patience} epochs -- stopping")
            break

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_dl, vocab, device, amp=use_amp,
                    holdout=set(cfg.holdout_cells))
    log(f"\ntest  SER {test['ser']:.4f}  WER {test['wer']:.4f}  "
        f"line-acc {test['line_accuracy']:.3f}")
    for style, s in test.get("by_style", {}).items():
        log(f"      {style:7s} SER {s['ser']:.4f}  ({s['n']} lines)")
    if "onset_error" in test:
        log(f"      onset err {test['onset_error']:.4f}  "
            f"vowel err {test['vowel_error']:.4f}")
    if "holdout_recall" in test:
        log(f"      held-out recall {test['holdout_recall']:.4f} "
            f"({test['holdout_tokens']} tokens)   "
            f"seen recall {test['seen_recall']:.4f} "
            f"({test['seen_tokens']} tokens)")

    result = {"config": cfg.as_dict(), "best_epoch": best_epoch,
              "best_val_ser": best, "test": test, "history": history,
              "n_parameters": count_parameters(model),
              "train_lines": len(train_ds),
              "minutes": (time.time() - t0) / 60.0}
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
