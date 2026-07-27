"""Legacy 36-skill contextual-bandit pilot.

This did not instantiate artificial childhood. It is preserved so the published
raw result remains reproducible and auditable.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
LABELS = ("<A>", "<B>", "<C>", "<D>")
SKILLS = [
    ("L1", 0), ("L2", 0), ("L3", 0), ("L4", 0), ("L5", 0), ("M1", 0),
    ("L6", 1), ("L7", 1), ("L8", 1), ("M2", 1), ("R1", 1), ("G1", 1),
    ("C1", 1), ("Q1", 1), ("M3", 2), ("R2", 2), ("R3", 2), ("S1", 2),
    ("S2", 2), ("S3", 2), ("G2", 2), ("C2", 2), ("M4", 3), ("M5", 3),
    ("R4", 3), ("S4", 3), ("S5", 3), ("G3", 3), ("G4", 3), ("C3", 3),
    ("C4", 3), ("M6", 4), ("R5", 4), ("C5", 4), ("C6", 4), ("Q2", 4),
]
STAGE_BY_SKILL = dict(SKILLS)
CONDITIONS = ("iid_clm", "ordered_clm", "adaptive_clm", "fixed_rl", "adaptive_rl", "adaptive_hybrid")


@dataclass
class Example:
    skill: str
    prompt: str
    answer: str
    visible: int = 0


def _choice(question: str, correct: str, wrong: list[str], rng: random.Random, skill: str) -> Example:
    values = [correct] + wrong
    rng.shuffle(values)
    answer = LABELS[values.index(correct)]
    options = " ".join(f"{LABELS[i]} {v}" for i, v in enumerate(values))
    return Example(skill, f"{question}\nChoices: {options}\nAnswer:", answer)


def _make_example(skill: str, seed: int, split: str = "train") -> Example:
    """Deterministic generators; far uses disjoint names, nouns, and phrasings."""
    rng = random.Random((seed + 1) * 1009 + sum(map(ord, skill)) * 917 + (0 if split == "train" else 10_000_019))
    train_names = ["Mira", "Lina", "Omar", "Tavi", "Niko", "Sela"]
    far_names = ["Zuri", "Keon", "Asha", "Bram", "Inez", "Yuki"]
    names = train_names if split == "train" else far_names
    a, b = rng.sample(names, 2)
    x, y = rng.randint(0, 5), rng.randint(0, 5)
    yes, no = "yes", "no"
    if skill == "L1":
        return _choice("The caregiver says: Your turn. Select the valid short reply.", "ready", ["later maybe", "???", "no reply"], rng, skill)
    if skill == "L2":
        same = rng.choice([True, False]); q = f"Are the symbols {'K and K' if same else 'K and P'} the same?"
        return _choice(q, yes if same else no, [no if same else yes, "unknown", "both"], rng, skill)
    if skill == "L3":
        truth = rng.choice([True, False]); q = f"A blue seed is in a pot. Is this statement true: the seed is {'blue' if truth else 'red'}?"
        return _choice(q, yes if truth else no, [no if truth else yes, "missing", "both"], rng, skill)
    if skill == "L4":
        return _choice(f"{a} carried a map. {b} carried a cup. Who carried the map?", a, [b, "both", "neither"], rng, skill)
    if skill == "L5":
        return _choice(f"The {a} stone is smooth. The {b} stone is rough. Which stone is rough?", b, [a, "both", "neither"], rng, skill)
    if skill == "M1":
        n = rng.randint(0, 5); items = " ".join(["seed"] * n) or "(empty)"
        return _choice(f"Count the seeds: {items}", str(n), [str((n + i) % 6) for i in (1, 2, 3)], rng, skill)
    if skill == "L6":
        return _choice(f"{a} arrived before {b}. Who arrived later?", b, [a, "both", "unknown"], rng, skill)
    if skill == "L7":
        return _choice(f"{a} met {b}. Then {a} picked up the key. Who picked up the key?", a, [b, "both", "unknown"], rng, skill)
    if skill == "L8":
        return _choice("Which instruction means the same as 'place the cup beside the box'?", "put the cup next to the box", ["put it inside", "remove the box", "move neither"], rng, skill)
    if skill == "M2":
        p, q = rng.sample(range(10), 2)
        return _choice(f"A jar has {p} beads and a tin has {q}. Which has more?", "jar" if p > q else "tin", ["tin" if p > q else "jar", "equal", "unknown"], rng, skill)
    if skill == "R1":
        u, v = rng.choice([True, False]), rng.choice([True, False]); op = rng.choice(["AND", "OR"])
        val = (u and v) if op == "AND" else (u or v)
        return _choice(f"P is {u}. Q is {v}. Is P {op} Q true?", yes if val else no, [no if val else yes, "unknown", "both"], rng, skill)
    if skill == "G1":
        return _choice("The robot faces north and turns right. Which way does it face?", "east", ["west", "north", "south"], rng, skill)
    if skill == "C1":
        return _choice("Start at 1. Apply +2, then +1. What is the final value?", "4", ["2", "3", "5"], rng, skill)
    if skill == "Q1":
        return _choice("How many seeds are in the closed box? No count was given.", "ask for the count", ["zero", "one", "ten"], rng, skill)
    if skill == "M3":
        total = x + y
        return _choice(f"{a} has {x} samples and receives {y}. How many now?", str(total), [str(total + 1), str(max(0, total - 1)), str(x)], rng, skill)
    if skill == "R2":
        wet = rng.choice([True, False]); val = wet
        return _choice(f"Rule: if soil is wet, the lamp is on. Soil is {'wet' if wet else 'not wet'}. Is the lamp guaranteed on?", yes if val else no, [no if val else yes, "both", "unknown"], rng, skill)
    if skill == "R3":
        return _choice("Claim 1: the gate is open. Claim 2: the gate is not open. Relationship?", "contradiction", ["agreement", "implication", "unrelated"], rng, skill)
    if skill == "S1":
        return _choice("Which is normally needed for a young plant to grow?", "water", ["plastic", "paint", "glass"], rng, skill)
    if skill == "S2":
        return _choice("Grass feeds rabbit; rabbit feeds fox. If rabbits vanish, what loses a food source?", "fox", ["grass", "sun", "soil"], rng, skill)
    if skill == "S3":
        return _choice("Liquid water is cooled below freezing. What state forms?", "solid", ["gas", "plasma", "unchanged liquid"], rng, skill)
    if skill == "G2":
        return _choice("Move north, then east. Where are you relative to the start?", "northeast", ["northwest", "southeast", "southwest"], rng, skill)
    if skill == "C2":
        return _choice("x = 2; then x = x + 3. What is x?", "5", ["2", "3", "6"], rng, skill)
    if skill == "M4":
        p, q = rng.randint(1, 5), rng.randint(1, 4); z = p * q
        return _choice(f"There are {p} groups of {q}. How many total?", str(z), [str(z + 1), str(p + q), str(max(0, z - q))], rng, skill)
    if skill == "M5":
        return _choice("Eight items are shared equally by four people. Each gets?", "2", ["1", "3", "4"], rng, skill)
    if skill == "R4":
        return _choice("To test whether salt harms growth, which action is an intervention?", "change salt while holding water fixed", ["watch only", "change every factor", "measure color once"], rng, skill)
    if skill == "S4":
        return _choice("How can sand be separated from water?", "filtering", ["melting", "freezing sand", "adding salt"], rng, skill)
    if skill == "S5":
        return _choice("A fair light experiment should change light and keep what fixed?", "water amount", ["all variables changing", "plant identity and water changing", "nothing"], rng, skill)
    if skill == "G3":
        return _choice("Which climate most supports a rainforest?", "warm and wet", ["cold and dry", "warm and dry", "cold and icy"], rng, skill)
    if skill == "G4":
        return _choice("A settlement needs drinking water. Which site is most suitable?", "near a clean river", ["salt flat", "dry ridge", "sealed cave"], rng, skill)
    if skill == "C3":
        return _choice("x=4. If x>3, set y=1; else y=0. What is y?", "1", ["0", "3", "4"], rng, skill)
    if skill == "C4":
        return _choice("Start x=0. Repeat x=x+2 three times. Final x?", "6", ["2", "3", "5"], rng, skill)
    if skill == "M6":
        n = rng.randint(1, 9); c = rng.randint(1, 5)
        return _choice(f"Unknown u satisfies u + {c} = {n+c}. What is u?", str(n), [str(n + c), str(c), str(max(0, n - 1))], rng, skill)
    if skill == "R5":
        return _choice("Rule: watering makes the lamp turn on. It was watered. Counterfactually, if it had not been watered, what follows?", "lamp not guaranteed on", ["lamp must stay on", "water doubles", "nothing can differ"], rng, skill)
    if skill == "C5":
        return _choice("Which definition creates a reusable procedure that adds one?", "function inc(x): return x+1", ["x+1 once", "print x", "delete x"], rng, skill)
    if skill == "C6":
        return _choice("Goal is 6. Trace: x=0; repeat twice: x=x+2. What is the bug?", "loop repeats too few times", ["addition too large", "x starts too high", "no bug"], rng, skill)
    return _choice("You computed 7, but a second independent check gives 6. Best next action?", "retrace the steps", ["ignore the check", "claim both", "stop checking"], rng, skill)


def make_example(skill: str, seed: int, split: str = "train") -> Example:
    example = _make_example(skill, seed, split)
    prefix = "Practice situation: " if split == "train" else "Unseen transfer situation: "
    example.prompt = prefix + example.prompt
    return example


class Tokenizer:
    def __init__(self, model_file: Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_file))
        self.pad = self.sp.pad_id()
        self.label_ids = torch.tensor([self.sp.piece_to_id(x) for x in LABELS])

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)


def build_tokenizer(path: Path) -> None:
    if path.exists():
        return
    corpus = ROOT / "artifacts" / "tokenizer_corpus.txt"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    with corpus.open("w", encoding="utf-8") as f:
        for split in ("train", "far"):
            for i in range(500):
                for skill, _ in SKILLS:
                    e = make_example(skill, i, split)
                    f.write(e.prompt + " " + e.answer + "\n")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(path.with_suffix("")), model_type="bpe",
        vocab_size=2048, byte_fallback=True, pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=list(LABELS), character_coverage=1.0,
    )


class Block(nn.Module):
    def __init__(self, d: int = 288, heads: int = 8, ff: int = 1152):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.gate, self.up, self.down = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)
        self.heads = heads
        self.register_buffer("rope_inv_freq", 1.0 / (10000 ** (torch.arange(0, d // heads, 2).float() / (d // heads))), persistent=False)

    def _rope(self, x):
        # x: batch, heads, time, head_dim
        angles = torch.outer(torch.arange(x.size(2), device=x.device), self.rope_inv_freq)
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        q, k, v = [z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v)]
        q, k = self._rope(q), self._rope(k)
        x = x + self.out(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Student(nn.Module):
    def __init__(self, vocab: int, layers: int = 8, d: int = 288):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(layers)])
        self.norm = nn.RMSNorm(d)
        self.lm = nn.Linear(d, vocab, bias=False)
        self.lm.weight = self.emb.weight
        self.value = nn.Linear(d, 1)

    def forward(self, ids):
        x = self.emb(ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm(x), self.value(x).squeeze(-1)


def batch_prompts(examples: list[Example], tok: Tokenizer, device):
    rows = [tok.encode(e.prompt) for e in examples]
    n = max(map(len, rows)); ids = torch.full((len(rows), n), tok.pad, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row); mask[i, len(row) - 1] = True
        examples[i].visible = len(row) + 1
    return ids.to(device), mask.to(device)


def batch_transcripts(examples: list[Example], tok: Tokenizer, device):
    rows = [tok.encode(e.prompt + " " + e.answer) for e in examples]
    n = max(map(len, rows)); ids = torch.full((len(rows), n), tok.pad, dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row); examples[i].visible = len(row)
    return ids.to(device)


class Scheduler:
    def __init__(self, adaptive: bool):
        self.adaptive = adaptive
        self.seen, self.correct = defaultdict(int), defaultdict(float)

    def sample(self, rng: random.Random, progress: float) -> str:
        open_stage = min(4, int(progress * 5.0))
        eligible = [s for s, stage in SKILLS if stage <= open_stage]
        if not self.adaptive:
            current = [s for s, stage in SKILLS if stage == open_stage]
            return rng.choice(current or eligible)
        weights = []
        for s in eligible:
            mastery = self.correct[s] / max(1, self.seen[s])
            weights.append(0.2 + (1.0 - mastery) + 0.35 * STAGE_BY_SKILL[s])
        return rng.choices(eligible, weights=weights, k=1)[0]

    def update(self, examples, rewards):
        for e, r in zip(examples, rewards):
            self.seen[e.skill] += 1
            self.correct[e.skill] += float(r)


@torch.no_grad()
def evaluate(model, tok, device, items_per_skill: int = 32):
    model.eval(); by_skill = {}
    label_ids = tok.label_ids.to(device)
    for skill, _ in SKILLS:
        examples = [make_example(skill, 900_000 + i, "far") for i in range(items_per_skill)]
        hits = 0
        for start in range(0, len(examples), 32):
            batch = examples[start:start + 32]; ids, mask = batch_prompts(batch, tok, device)
            logits, _ = model(ids); last = logits[mask][:, label_ids]
            pred = last.argmax(-1).cpu().tolist()
            truth = [LABELS.index(e.answer) for e in batch]
            hits += sum(a == b for a, b in zip(pred, truth))
        by_skill[skill] = hits / len(examples)
    model.train()
    groups = {
        "language": [s for s, _ in SKILLS if s.startswith(("L", "Q"))],
        "mathematics": [s for s, _ in SKILLS if s.startswith("M")],
        "reasoning": [s for s, _ in SKILLS if s.startswith("R")],
        "science_geography": [s for s, _ in SKILLS if s.startswith(("S", "G"))],
        "coding": [s for s, _ in SKILLS if s.startswith("C")],
    }
    return {"macro": float(np.mean(list(by_skill.values()))),
            "groups": {g: float(np.mean([by_skill[s] for s in ss])) for g, ss in groups.items()},
            "skills": by_skill}


def train(condition: str, seed: int, budget: int, tok: Tokenizer, device, layers: int, d: int):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = Student(tok.sp.vocab_size(), layers, d).to(device)
    params = sum(p.numel() for p in model.parameters())
    lr = 3e-4 if "clm" in condition else 1e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    adaptive = condition in ("adaptive_clm", "adaptive_rl", "adaptive_hybrid")
    scheduler = Scheduler(adaptive); rng = random.Random(seed + 77)
    visible = steps = 0; losses = []; started = time.perf_counter()
    label_ids = tok.label_ids.to(device)
    while visible < budget:
        progress = visible / budget
        examples = [make_example(scheduler.sample(rng, progress), rng.randrange(1 << 30)) for _ in range(16)]
        if condition == "iid_clm":
            examples = [make_example(rng.choice(SKILLS)[0], rng.randrange(1 << 30)) for _ in examples]
        opt.zero_grad(set_to_none=True)
        if condition in ("iid_clm", "ordered_clm", "adaptive_clm"):
            ids = batch_transcripts(examples, tok, device)
            logits, _ = model(ids[:, :-1])
            targets = ids[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=tok.pad)
            loss.backward()
            rewards = [0.0] * len(examples)
        else:
            ids, mask = batch_prompts(examples, tok, device)
            logits, values = model(ids)
            policy = logits[mask][:, label_ids]
            dist = torch.distributions.Categorical(logits=policy)
            actions = dist.sample()
            truth = torch.tensor([LABELS.index(e.answer) for e in examples], device=device)
            reward = (actions == truth).float()
            advantage = reward - values[mask].detach()
            policy_loss = -(dist.log_prob(actions) * advantage).mean()
            value_loss = F.mse_loss(values[mask], reward)
            entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            if condition == "adaptive_hybrid":
                full = batch_transcripts(examples, tok, device)
                lm_logits, _ = model(full[:, :-1])
                clm = F.cross_entropy(lm_logits.reshape(-1, lm_logits.size(-1)), full[:, 1:].reshape(-1), ignore_index=tok.pad)
                loss = loss + 0.10 * clm
            loss.backward()
            rewards = reward.detach().cpu().tolist()
            scheduler.update(examples, rewards)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); losses.append(float(loss.detach()))
        visible += sum(e.visible for e in examples); steps += 1
    metrics = evaluate(model, tok, device)
    metrics.update({"condition": condition, "seed": seed, "visible_tokens": visible, "optimizer_steps": steps,
                    "wall_seconds": time.perf_counter() - started, "parameters": params,
                    "mean_final_20_loss": float(np.mean(losses[-20:])),
                    "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0})
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=50_000, help="visible tokens per condition")
    p.add_argument("--seeds", type=int, nargs="+", default=[1000])
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--items-per-skill", type=int, default=32)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--hidden", type=int, default=288)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    model_file = ROOT / "artifacts" / "tokenizer.model"; build_tokenizer(model_file)
    tok = Tokenizer(model_file); device = torch.device(args.device)
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    results = []
    for seed in args.seeds:
        for condition in args.conditions:
            if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
            print(f"running {condition} seed={seed} budget={args.budget}", flush=True)
            result = train(condition, seed, args.budget, tok, device, args.layers, args.hidden)
            results.append(result)
            (out / f"{run_id}.json").write_text(json.dumps({"config": vars(args), "results": results}, indent=2), encoding="utf-8")
            print(f"  far-transfer macro={result['macro']:.3f} time={result['wall_seconds']:.1f}s", flush=True)
    summary = {"run_id": run_id, "status": "pilot_not_confirmatory", "config": vars(args), "results": results}
    (out / f"{run_id}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out / f"{run_id}.json")


if __name__ == "__main__":
    main()
