"""CRNN with two interchangeable output heads.

The backbone is deliberately the standard CRNN: convolutional feature
extractor, height collapsed to one, bidirectional LSTM over the width axis,
CTC on top.  Keeping it ordinary is the point -- the comparison under study is
the *label space*, so the encoder must not differ between conditions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .vocab import Vocab


class ConvBackbone(nn.Module):
    """Collapses a fixed-height line image to a width-indexed feature sequence.

    Height is reduced by 32 (so a 64px line becomes 2 rows, then pooled to 1);
    width by 4, which leaves ~25 timesteps per 100px of line -- comfortably
    more than the number of syllables CTC has to emit.
    """

    def __init__(self, in_ch: int = 1, out_dim: int = 512):
        super().__init__()

        def block(i, o, pool=None, bn=True):
            layers = [nn.Conv2d(i, o, 3, 1, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(o))
            layers.append(nn.ReLU(inplace=True))
            if pool is not None:
                layers.append(nn.MaxPool2d(pool))
            return layers

        self.net = nn.Sequential(
            *block(in_ch, 64, pool=(2, 2)),      # H/2  W/2
            *block(64, 128, pool=(2, 2)),        # H/4  W/4
            *block(128, 256),
            *block(256, 256, pool=(2, 1)),       # H/8  W/4
            *block(256, 512),
            *block(512, 512, pool=(2, 1)),       # H/16 W/4
            *block(512, out_dim, pool=(2, 1)),   # H/32 W/4
        )
        self.out_dim = out_dim
        self.width_stride = 4

    def forward(self, x):                       # (B,1,H,W)
        f = self.net(x)                         # (B,C,H',W')
        f = f.mean(dim=2)                       # collapse height -> (B,C,W')
        return f.permute(0, 2, 1).contiguous()  # (B,W',C)


class PlainHead(nn.Module):
    """One free parameter vector per class."""

    def __init__(self, dim: int, vocab: Vocab):
        super().__init__()
        self.proj = nn.Linear(dim, vocab.n_classes)

    def forward(self, h):
        return self.proj(h)


class FactoredHead(nn.Module):
    """Syllable score = onset score + vowel score.

    A consonant's parameters are trained by every vowel column it appears in,
    and a vowel sign's by every consonant it attaches to.  Crucially there is
    no per-syllable free parameter, so a syllable never seen in training still
    receives a well-defined score -- which is what makes held-out (onset,
    vowel) cells answerable rather than impossible by construction.

    Blank and the word separator are not syllables and keep their own vectors.
    """

    def __init__(self, dim: int, vocab: Vocab):
        super().__init__()
        self.n_classes = vocab.n_classes
        self.onset_proj = nn.Linear(dim, len(vocab.onsets))
        self.vowel_proj = nn.Linear(dim, len(vocab.vowels))
        n_special = 1 + vocab.n_specials            # blank + <sp>
        self.special_proj = nn.Linear(dim, n_special)

        onset_of = torch.tensor(vocab.onset_of, dtype=torch.long)
        vowel_of = torch.tensor(vocab.vowel_of, dtype=torch.long)
        special_of = torch.tensor(vocab.special_of, dtype=torch.long)
        # blank occupies special slot 0; <sp> and friends follow
        special_slot = torch.where(special_of >= 0, special_of + 1,
                                   torch.full_like(special_of, -1))
        special_slot[0] = 0
        is_syllable = onset_of >= 0

        self.register_buffer("onset_of", onset_of.clamp(min=0))
        self.register_buffer("vowel_of", vowel_of.clamp(min=0))
        self.register_buffer("special_slot", special_slot.clamp(min=0))
        self.register_buffer("is_syllable", is_syllable)

    def forward(self, h):                        # (B,T,D)
        onset = self.onset_proj(h)               # (B,T,n_onsets)
        vowel = self.vowel_proj(h)               # (B,T,n_vowels)
        special = self.special_proj(h)           # (B,T,n_special)

        syl = (onset[..., self.onset_of] + vowel[..., self.vowel_of])
        spc = special[..., self.special_slot]
        return torch.where(self.is_syllable, syl, spc)

    def factor_logits(self, h):
        """Onset and vowel logits on their own, for per-factor analysis."""
        return self.onset_proj(h), self.vowel_proj(h)


class CRNN(nn.Module):
    def __init__(self, vocab: Vocab, head: str = "plain",
                 conv_dim: int = 512, rnn_hidden: int = 256,
                 rnn_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.vocab = vocab
        self.backbone = ConvBackbone(1, conv_dim)
        self.rnn = nn.LSTM(conv_dim, rnn_hidden, num_layers=rnn_layers,
                           bidirectional=True, batch_first=True,
                           dropout=dropout if rnn_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)

        dim = 2 * rnn_hidden
        if head == "plain":
            self.head = PlainHead(dim, vocab)
        elif head == "factored":
            self.head = FactoredHead(dim, vocab)
        else:
            raise ValueError(f"unknown head {head!r}; use 'plain' or 'factored'")
        self.head_name = head

    @property
    def width_stride(self) -> int:
        return self.backbone.width_stride

    def input_lengths(self, widths) -> torch.Tensor:
        return torch.clamp(widths // self.width_stride, min=1)

    def forward(self, images):
        f = self.backbone(images)         # (B,T,C)
        f, _ = self.rnn(f)
        f = self.dropout(f)
        return self.head(f)               # (B,T,n_classes)


def build_model(vocab: Vocab, head: str = "plain", **kw) -> CRNN:
    return CRNN(vocab, head=head, **kw)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
