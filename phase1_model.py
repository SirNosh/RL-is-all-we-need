from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase1_data import Tokenizer

SCORE_CONTEXT_BATCH = 16


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.gate, self.up, self.down = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)
        self.heads = heads
        hd = d // heads
        self.register_buffer("rope_inv_freq", 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd)), persistent=False)

    def _rope(self, x):
        angles = torch.outer(torch.arange(x.size(2), device=x.device), self.rope_inv_freq)
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        q, k, v = [z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(self._rope(q), self._rope(k), v, is_causal=True)
        x = x + self.out(y.transpose(1, 2).reshape(b, t, d))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Student(nn.Module):
    def __init__(self, vocab: int, layers=8, d=288, heads=8, ff=1152):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        self.norm, self.lm, self.value = nn.RMSNorm(d), nn.Linear(d, vocab, bias=False), nn.Linear(d, 1)
        self.lm.weight = self.emb.weight

    def forward(self, ids):
        x = self.emb(ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm(x), self.value(x).squeeze(-1)


def candidate_scores(model: Student, tok: Tokenizer, contexts: Sequence[str], candidates: Sequence[str], device):
    if len(contexts) > SCORE_CONTEXT_BATCH:
        chunks = [candidate_scores(model, tok, contexts[i:i+SCORE_CONTEXT_BATCH], candidates, device) for i in range(0, len(contexts), SCORE_CONTEXT_BATCH)]
        return torch.cat([x[0] for x in chunks]), torch.cat([x[1] for x in chunks])
    encoded = [tok.encode(" " + c) + [tok.eos] for c in candidates]
    context_ids = [tok.encode(c) for c in contexts]
    rows, starts = [], []
    for prefix in context_ids:
        for response in encoded:
            rows.append(prefix + response); starts.append(len(prefix))
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), tok.pad, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, values = model(ids)
    logp = logits.log_softmax(-1)
    scores = torch.empty((len(contexts), len(candidates)), device=device)
    context_values = torch.empty(len(contexts), device=device)
    cursor = 0
    for owner, prefix in enumerate(context_ids):
        for j, response in enumerate(encoded):
            start = starts[cursor]
            pos = torch.arange(start - 1, start + len(response) - 1, device=device)
            targets = ids[cursor, start:start+len(response)]
            scores[owner, j] = logp[cursor, pos, targets].sum()
            if j == 0:
                context_values[owner] = values[cursor, start - 1]
            cursor += 1
    return scores, context_values


def clm_loss(model: Student, tok: Tokenizer, texts: Sequence[str], device):
    rows = [tok.encode(x) + [tok.eos] for x in texts]
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), tok.pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row, device=device); mask[i, :len(row)] = True
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, _ = model(ids[:, :-1])
        losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), reduction="none")
    valid = mask[:, 1:].reshape(-1)
    return (losses * valid).sum() / valid.sum().clamp_min(1)
