"""Stage-0 Textual Bootstrap Study.

The policy emits short textual replies selected autoregressively from a
skill-appropriate response grammar. This keeps cold-start exploration finite
without encoding which reply is correct. The grammar constraint is explicit:
unconstrained byte-level production is a later ablation, not silently claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

from legacy_pilot import Student

ROOT = Path(__file__).resolve().parent
SCORING_BATCH_SIZE = 32
SKILLS = ("turn", "truth", "entity", "property", "count", "clarify")
PREREQUISITES = {
    "turn": (), "truth": ("turn",), "entity": ("turn",),
    "property": ("entity",), "count": ("entity",), "clarify": ("truth", "entity"),
}
DEFAULT_CONDITIONS = ("iid_clm", "ordered_clm", "fixed_caregiver_rl",
                      "adaptive_caregiver_rl", "adaptive_hybrid")
CONDITIONS = DEFAULT_CONDITIONS + ("adaptive_clm",)


@dataclass
class Situation:
    skill: str
    prompt: str
    candidates: list[str]
    answer: int
    explanation: str


@dataclass
class Transition:
    dialogue: int
    skill: str
    context: str
    candidates: list[str]
    action: int
    reward: float
    old_logp: float
    old_value: float
    advantage: float = 0.0
    return_: float = 0.0


def _rng(skill: str, seed: int, family: int) -> random.Random:
    return random.Random(seed * 10007 + family * 1_000_003 + sum(map(ord, skill)) * 97)


def make_situation(skill: str, seed: int, family: int) -> Situation:
    """Eight independently worded families: train 0-3, near 4-5, far 6-7."""
    r = _rng(skill, seed, family)
    train_names = ("Mira", "Niko", "Sela", "Tavi", "Omar", "Lina")
    eval_names = ("Asha", "Bram", "Inez", "Keon", "Yuki", "Zuri")
    names = train_names if family < 4 else eval_names
    a, b, c = r.sample(names, 3)

    if skill == "turn":
        prompts = [
            "Your turn. Give the short reply that shows you are ready.",
            "I have finished speaking. Respond when you are ready.",
            "The caregiver asks whether the lesson may begin. Reply briefly.",
            "Take your turn now and signal readiness.",
            "What short response appropriately begins your turn?",
            "The speaker pauses for you. Indicate that you can continue.",
            "After listening to the instructions, what can you say to begin?",
            "A conversation partner yields the floor. Respond appropriately.",
        ]
        candidates = ["ready", "not ready", "please tell me", "yes"]
        return Situation(skill, prompts[family], candidates, 0, "A short readiness signal is 'ready'.")

    if skill == "truth":
        color, other = r.choice([("blue", "red"), ("green", "yellow")])
        true = r.choice([True, False]); stated = color if true else other
        prompts = [
            f"A seed is {color}. Is the seed {stated}?",
            f"The observed seed has the color {color}. Answer yes or no: it is {stated}.",
            f"Fact: seed color = {color}. Claim: seed color = {stated}. Is the claim true?",
            f"We painted one seed {color}. Would it be correct to call that seed {stated}?",
            f"Decide whether the following matches the record. Record: {color}. Statement: {stated}.",
            f"The word on the seed's label is {color}; someone says {stated}. Are they right?",
            f"Although {b} discusses a red pot, the seed itself is {color}. Is the seed {stated}?",
            f"Is '{stated}' a truthful answer to 'what color is the seed?' The seed is {color}.",
        ]
        return Situation(skill, prompts[family], ["yes", "no"], 0 if true else 1,
                         f"The seed is {color}, so the answer is {'yes' if true else 'no'}.")

    if skill == "entity":
        objects = r.sample(("cup", "key", "book", "bowl", "map", "ring"), 3)
        prompts = [
            f"{a} holds the {objects[0]}. {b} holds the {objects[1]}. Who holds the {objects[0]}?",
            f"The {objects[1]} belongs to {b}, while {a} is carrying the {objects[0]}. Name the {objects[0]}'s holder.",
            f"Record: {a} -> {objects[0]}; {b} -> {objects[1]}. Query: holder of {objects[0]}?",
            f"I gave {objects[0]} to {a} and {objects[1]} to {b}. Which person received {objects[0]}?",
            f"Who has {objects[0]}? We know {b} has {objects[1]} and {a} has {objects[0]}.",
            f"Two possessions were observed. The {objects[0]} was with {a}; the {objects[1]} was with {b}. Answer with a name.",
            f"{c} watches quietly. {b}, not {a}, holds {objects[1]}. The remaining fact says {a} holds {objects[0]}. Who has {objects[0]}?",
            f"Asked about {objects[0]}, ignore the later fact that {b} owns {objects[1]}; earlier we learned its carrier was {a}. Who?",
        ]
        candidates = list(names)
        return Situation(skill, prompts[family], candidates, candidates.index(a),
                         f"{a} holds the {objects[0]}; the answer is {a}.")

    if skill == "property":
        item1, item2 = r.sample(("stone", "leaf", "tile", "shell"), 2)
        prop, other = r.choice([("smooth", "rough"), ("warm", "cold"), ("large", "small")])
        prompts = [
            f"The {item1} is {prop}. The {item2} is {other}. Which object is {prop}?",
            f"One object is {other}: the {item2}. In contrast, the {item1} is {prop}. Name the {prop} object.",
            f"Properties: {item1}={prop}; {item2}={other}. Query: what is {prop}?",
            f"Touching them reveals a {prop} {item1} and a {other} {item2}. Which is {prop}?",
            f"Which object has the property '{prop}'? The {item2} is {other}, but the {item1} is {prop}.",
            f"Choose the noun bound to {prop}, given that {item1} has it and {item2} has {other}.",
            f"{a} mentions the {item2}, which is {other}. Separately, the {item1} is described as {prop}. What is {prop}?",
            f"The relevant property is {prop}. It belongs to {item1}; {item2}, mentioned first, is {other}. Answer with the object.",
        ]
        candidates = [item1, item2, "both", "neither"]
        return Situation(skill, prompts[family], candidates, 0, f"The {item1} is {prop}.")

    if skill == "count":
        n = r.randint(0, 5); noun = r.choice(("seeds", "beads", "cups"))
        words = ("zero", "one", "two", "three", "four", "five")
        marks = " ".join(["*" for _ in range(n)]) or "(none)"
        prompts = [
            f"Count the {noun}: {marks}",
            f"How many {noun} are shown here? {marks}",
            f"Inventory marks for {noun}: {marks}. Give the count as a word.",
            f"Each star is one {noun[:-1]}. Stars: {marks}. How many?",
            f"State the number of {noun} represented by: {marks}",
            f"Without adding anything, count this collection of {noun}: {marks}",
            f"{a} sees {marks}, where every star denotes exactly one {noun[:-1]}. Ignore {b}'s empty bag. How many are present?",
            f"Answer in a number word. The complete set of {noun}, after irrelevant labels are removed, is {marks}.",
        ]
        return Situation(skill, prompts[family], list(words), n, f"There are {words[n]} {noun}.")

    prompts = [
        "A closed box contains some seeds, but no count was given. How many are there?",
        "You were asked for the number of cups. The description omits the number. What should you request?",
        "Known facts list a box but no quantity. Respond appropriately instead of guessing.",
        "The caregiver asks for an absent count. What useful clarification should you make?",
        "Can the quantity be answered from a statement that supplies no quantity? Give the useful response.",
        "Rather than inventing a number, request the missing count.",
        f"{a} knows the box color and {b} knows its owner, but nobody states its quantity. What should you say?",
        "Several irrelevant properties are supplied, yet the requested number is absent. Respond with a clarification.",
    ]
    candidates = ["please tell me the count", "zero", "one", "five"]
    return Situation(skill, prompts[family], candidates, 0,
                     "The quantity is missing, so ask: please tell me the count.")


class Tokenizer:
    def __init__(self, model: Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model))
        self.pad, self.eos = self.sp.pad_id(), self.sp.eos_id()

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)


def build_tokenizer(path: Path) -> None:
    if path.exists():
        return
    corpus = path.with_name("stage0_tokenizer_corpus.txt")
    corpus.parent.mkdir(parents=True, exist_ok=True)
    with corpus.open("w", encoding="utf-8") as f:
        for seed in range(1500):
            for family in range(4):  # Training families only. Never evaluation.
                for skill in SKILLS:
                    s = make_situation(skill, seed, family)
                    f.write(s.prompt + "\n" + s.explanation + "\n")
                    for candidate in s.candidates:
                        f.write(candidate + "\n")
        # Ensure the locked 2,048-piece BPE can be formed without borrowing
        # evaluation text. These strings carry no task facts or hidden wording.
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        for a in alphabet:
            for b in alphabet:
                f.write(f"tokenizer reserve {a}{b}\n")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(path.with_suffix("")), model_type="bpe",
        vocab_size=2048, byte_fallback=True, pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        character_coverage=1.0,
    )


def _candidate_scores(model, tok: Tokenizer, contexts: list[str],
                      candidate_lists: list[list[str]], device):
    """Differentiable log P(candidate + EOS | context), plus context value."""
    if len(contexts) > SCORING_BATCH_SIZE:
        chunks = [_candidate_scores(model, tok, contexts[i:i + SCORING_BATCH_SIZE],
                                    candidate_lists[i:i + SCORING_BATCH_SIZE], device)
                  for i in range(0, len(contexts), SCORING_BATCH_SIZE)]
        width = max(scores.size(1) for scores, _ in chunks)
        padded = [F.pad(scores, (0, width - scores.size(1)), value=-1e9)
                  for scores, _ in chunks]
        return torch.cat(padded), torch.cat([values for _, values in chunks])
    encoded_candidates = [[tok.encode(" " + candidate) for candidate in candidates]
                          for candidates in candidate_lists]
    if all(len(piece_ids) == 1 for candidates in encoded_candidates for piece_ids in candidates):
        context_ids = [tok.encode(c) for c in contexts]
        width = max(map(len, context_ids))
        ids = torch.full((len(contexts), width), tok.pad, dtype=torch.long, device=device)
        for i, row in enumerate(context_ids):
            ids[i, :len(row)] = torch.tensor(row, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits, values = model(ids)
        max_candidates = max(map(len, candidate_lists))
        scores = torch.full((len(contexts), max_candidates), -1e9, device=device)
        result_values = torch.empty(len(contexts), device=device)
        for i, candidate_ids in enumerate(encoded_candidates):
            last_logits = logits[i, len(context_ids[i]) - 1].log_softmax(-1)
            ids_for_candidates = torch.tensor([x[0] for x in candidate_ids], device=device)
            scores[i, :len(candidate_ids)] = last_logits[ids_for_candidates]
            result_values[i] = values[i, len(context_ids[i]) - 1]
        return scores, result_values
    rows, owners, starts = [], [], []
    context_ids = [tok.encode(c) for c in contexts]
    for i, candidates in enumerate(candidate_lists):
        for candidate in candidates:
            prefix = context_ids[i]
            response = tok.encode(" " + candidate) + [tok.eos]
            rows.append(prefix + response); owners.append(i); starts.append(len(prefix))
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), tok.pad, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        logits, values = model(ids[:, :-1])
    log_probs = logits.log_softmax(-1)
    flat_scores = []
    for i, row in enumerate(rows):
        start = starts[i]
        targets = ids[i, start:len(row)]
        positions = torch.arange(start - 1, len(row) - 1, device=device)
        flat_scores.append(log_probs[i, positions, targets].sum())
    max_candidates = max(map(len, candidate_lists))
    scores = torch.full((len(contexts), max_candidates), -1e9, device=device)
    vals = torch.empty(len(contexts), device=device)
    cursor = 0
    for i, candidates in enumerate(candidate_lists):
        for j in range(len(candidates)):
            scores[i, j] = flat_scores[cursor]; cursor += 1
        first_row = sum(len(x) for x in candidate_lists[:i])
        vals[i] = values[first_row, len(context_ids[i]) - 1]
    return scores, vals


class Curriculum:
    def __init__(self, adaptive: bool):
        self.adaptive = adaptive
        self.mastered: set[str] = set()
        self.streak = defaultdict(int)
        self.diagnostics: list[dict] = []
        self.skill_selections = defaultdict(int)
        self.review_selections = 0
        self.total_selections = 0

    def update_diagnostics(self, scores: dict[str, dict[str, float]], tokens: int):
        for skill in SKILLS:
            passed = scores[skill]["independent"] >= .80 and scores[skill]["far"] >= .70
            self.streak[skill] = self.streak[skill] + 1 if passed else 0
            if self.streak[skill] >= 2:
                self.mastered.add(skill)
        self.diagnostics.append({"tokens": tokens, "scores": scores, "mastered": sorted(self.mastered)})

    def eligible(self):
        if not self.adaptive:
            return list(SKILLS)
        result = [s for s in SKILLS if all(p in self.mastered for p in PREREQUISITES[s])]
        return result or ["turn"]

    def sample(self, rng: random.Random, progress: float) -> str:
        if self.adaptive:
            eligible = self.eligible()
            review = [s for s in eligible if s in self.mastered]
            learning = [s for s in eligible if s not in self.mastered]
            if review and rng.random() < .20:
                selected = rng.choice(review)
                self.review_selections += 1
            else:
                selected = rng.choice(learning or eligible)
            self.skill_selections[selected] += 1
            self.total_selections += 1
            return selected
        current = min(len(SKILLS) - 1, int(progress * len(SKILLS)))
        if current and rng.random() < .20:
            selected = rng.choice(SKILLS[:current])
            self.review_selections += 1
        else:
            selected = SKILLS[current]
        self.skill_selections[selected] += 1
        self.total_selections += 1
        return selected

    def state_dict(self):
        return {
            "adaptive": self.adaptive, "mastered": sorted(self.mastered),
            "streak": dict(self.streak), "diagnostics": self.diagnostics,
            "skill_selections": dict(self.skill_selections),
            "review_selections": self.review_selections,
            "total_selections": self.total_selections,
        }

    def load_state_dict(self, state):
        self.mastered = set(state["mastered"])
        self.streak = defaultdict(int, state["streak"])
        self.diagnostics = state["diagnostics"]
        self.skill_selections = defaultdict(int, state["skill_selections"])
        self.review_selections = state["review_selections"]
        self.total_selections = state["total_selections"]


@torch.no_grad()
def accuracy(model, tok, device, skill: str, families: tuple[int, ...],
             count: int, seeds: list[int] | None = None):
    model.eval(); hits = total = 0
    seeds = seeds or list(range(800_000, 800_000 + count))
    situations = [make_situation(skill, seed, families[i % len(families)]) for i, seed in enumerate(seeds)]
    for start in range(0, len(situations), 64):
        batch = situations[start:start + 64]
        contexts = ["Caregiver: " + s.prompt + "\nChild:" for s in batch]
        scores, _ = _candidate_scores(model, tok, contexts, [s.candidates for s in batch], device)
        pred = scores.argmax(-1).cpu().tolist()
        hits += sum(p == s.answer for p, s in zip(pred, batch)); total += len(batch)
    model.train()
    return hits / total


@torch.no_grad()
def diagnostics(model, tok, device, count=128):
    return {skill: {
        "independent": accuracy(model, tok, device, skill, (4, 5), count),
        "far": accuracy(model, tok, device, skill, (6, 7), count),
    } for skill in SKILLS}


def collect_rollout(model, tok, device, curriculum, rng, dialogues, progress,
                    fixed_skill=None, fixed_seeds=None):
    model.eval(); transitions: list[Transition] = []; started = time.perf_counter()
    situations = []; dialogue_ids = []
    for i in range(dialogues):
        skill = fixed_skill or curriculum.sample(rng, progress)
        seed = fixed_seeds[i % len(fixed_seeds)] if fixed_seeds else rng.randrange(1 << 30)
        family = rng.randrange(4)
        situations.append(make_situation(skill, seed, family))
        dialogue_ids.append(f"{skill}:{seed}:{family}")
    active = list(range(dialogues))
    contexts = ["Caregiver: " + s.prompt + "\nChild:" for s in situations]
    visible = sum(len(tok.encode(c)) for c in contexts)
    for attempt in range(3):
        if not active:
            break
        batch_contexts = [contexts[i] for i in active]
        batch_candidates = [situations[i].candidates for i in active]
        with torch.no_grad():
            scores, values = _candidate_scores(model, tok, batch_contexts, batch_candidates, device)
            dist = torch.distributions.Categorical(logits=scores)
            actions = dist.sample(); logps = dist.log_prob(actions)
        next_active = []
        for j, dialogue in enumerate(active):
            s = situations[dialogue]; action = int(actions[j])
            correct = action == s.answer
            reward = (1.0, .6, .3)[attempt] if correct else -.25
            reply = s.candidates[action]
            transitions.append(Transition(dialogue, s.skill, contexts[dialogue],
                                          s.candidates, action, reward,
                                          float(logps[j]), float(values[j])))
            visible += len(tok.encode(" " + reply))
            if not correct and attempt < 2:
                feedback = s.explanation if attempt == 0 else "Worked example: " + s.explanation
                analogous = make_situation(s.skill, rng.randrange(1 << 30), rng.randrange(4))
                contexts[dialogue] += f" {reply}\nCaregiver: Not quite. {feedback}\nCaregiver: {analogous.prompt}\nChild:"
                situations[dialogue] = analogous
                visible += len(tok.encode(feedback + analogous.prompt))
                next_active.append(dialogue)
        active = next_active
    by_dialogue = defaultdict(list)
    for t in transitions:
        by_dialogue[t.dialogue].append(t)
    for sequence in by_dialogue.values():
        gae = 0.0; next_value = 0.0
        for t in reversed(sequence):
            delta = t.reward + .99 * next_value - t.old_value
            gae = delta + .99 * .95 * gae
            t.advantage = gae; t.return_ = gae + t.old_value
            next_value = t.old_value
    model.train()
    skill_rewards, skill_counts, action_counts = defaultdict(float), defaultdict(int), defaultdict(lambda: defaultdict(int))
    first_attempts = [sequence[0] for sequence in by_dialogue.values()]
    for t in transitions:
        skill_rewards[t.skill] += t.reward
        skill_counts[t.skill] += 1
        action_counts[t.skill][t.candidates[t.action]] += 1
    metrics = {
        "collection_seconds": time.perf_counter() - started,
        "dialogues": dialogues,
        "transitions": len(transitions),
        "retries": len(transitions) - dialogues,
        "attempt1_accuracy": float(np.mean([t.reward > 0 for t in first_attempts])),
        "eventual_success": float(np.mean([any(t.reward > 0 for t in sequence) for sequence in by_dialogue.values()])),
        "invalid_response_rate": 0.0,
        "mean_reward_by_skill": {s: skill_rewards[s] / skill_counts[s] for s in skill_counts},
        "transitions_by_skill": dict(skill_counts),
        "candidate_distribution": {s: dict(counts) for s, counts in action_counts.items()},
        "dialogue_ids_hash": _sha256_bytes("\n".join(dialogue_ids).encode()),
    }
    return transitions, visible, metrics


def ppo_update(model, tok, device, transitions, optimizer, rng, target_kl=.03):
    by_skill = defaultdict(list)
    for i, t in enumerate(transitions):
        by_skill[t.skill].append(i)
    advantages = torch.tensor([t.advantage for t in transitions])
    for indices in by_skill.values():
        v = advantages[indices]
        advantages[indices] = (v - v.mean()) / (v.std(unbiased=False) + 1e-8)
    logs = defaultdict(list)
    early_stopped = False; updates = 0; epochs_completed = 0
    for epoch in range(2):
        order = list(range(len(transitions))); rng.shuffle(order)
        for start in range(0, len(order), 64):
            idx = order[start:start + 64]; batch = [transitions[i] for i in idx]
            scores, values = _candidate_scores(model, tok, [t.context for t in batch],
                                                [t.candidates for t in batch], device)
            dist = torch.distributions.Categorical(logits=scores)
            actions = torch.tensor([t.action for t in batch], device=device)
            new_logp = dist.log_prob(actions)
            old_logp = torch.tensor([t.old_logp for t in batch], device=device)
            adv = advantages[idx].to(device)
            ratio = (new_logp - old_logp).exp()
            log_ratio = new_logp - old_logp
            approx_kl = ((ratio - 1) - log_ratio).mean()
            if updates and float(approx_kl.detach()) > 1.5 * target_kl:
                logs["kl"].append(float(approx_kl.detach()))
                early_stopped = True
                break
            clipped = ratio.clamp(.8, 1.2)
            policy_loss = -torch.minimum(ratio * adv, clipped * adv).mean()
            returns = torch.tensor([t.return_ for t in batch], device=device)
            value_loss = F.mse_loss(values, returns)
            entropy = dist.entropy().mean()
            loss = policy_loss + .5 * value_loss - .02 * entropy
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            updates += 1
            logs["kl"].append(float(approx_kl.detach()))
            logs["clip_fraction"].append(float(((ratio - 1).abs() > .2).float().mean().detach()))
            logs["entropy"].append(float(entropy.detach()))
            logs["value_error"].append(float(value_loss.detach()))
        if early_stopped:
            break
        epochs_completed = epoch + 1
    old_values = np.array([t.old_value for t in transitions])
    returns = np.array([t.return_ for t in transitions])
    explained = 1 - np.var(returns - old_values) / max(np.var(returns), 1e-8)
    return {k: float(np.mean(v)) for k, v in logs.items()} | {
        "explained_variance": float(explained), "early_stopped_for_kl": early_stopped,
        "optimizer_minibatches": updates, "epochs_completed": epochs_completed,
    }


def clm_update(model, tok, device, situations, optimizer, weight=1.0):
    texts = [f"Caregiver: {s.prompt}\nChild: {s.candidates[s.answer]}\nCaregiver: Correct. {s.explanation}" for s in situations]
    rows = [tok.encode(x) + [tok.eos] for x in texts]
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), tok.pad, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row, device=device)
    logits, _ = model(ids[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), ignore_index=tok.pad)
    optimizer.zero_grad(set_to_none=True); (weight * loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    return float(loss.detach()), sum(map(len, rows))


def make_model(tok, device, seed, layers=8, hidden=288, policy_lr=1e-4):
    torch.manual_seed(seed)
    model = Student(tok.sp.vocab_size(), layers, hidden).to(device)
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
    model.apply(initialize)
    value_params = list(model.value.parameters())
    value_ids = {id(p) for p in value_params}
    policy_params = [p for p in model.parameters() if id(p) not in value_ids]
    optimizer = torch.optim.AdamW([
        {"params": policy_params, "lr": policy_lr},
        {"params": value_params, "lr": 3e-4},
    ], betas=(.9, .95), weight_decay=.1)
    return model, optimizer


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _config_dict(args):
    excluded = {"resume", "max_rollouts", "overfit_test", "overfit_rollouts"}
    return {k: v for k, v in vars(args).items() if k not in excluded}


def _config_hash(args) -> str:
    return _sha256_bytes(json.dumps(_config_dict(args), sort_keys=True).encode())


def _code_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], cwd=ROOT,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_json(path: Path, payload) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)


def save_checkpoint(path: Path, model, optimizer, scheduler, visible, rollout,
                    curriculum, rng, condition, seed, tok_path, args):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "visible_tokens": visible,
        "rollout": rollout,
        "curriculum": curriculum.state_dict(),
        "skill_selection_state": curriculum.state_dict(),
        "diagnostic_history": curriculum.diagnostics,
        "python_rng": random.getstate(),
        "local_python_rng": rng.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "condition": condition,
        "seed": seed,
        "tokenizer_hash": _file_hash(tok_path),
        "code_commit": _code_commit(),
        "configuration": _config_dict(args),
        "configuration_hash": _config_hash(args),
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, curriculum, rng,
                    tok_path: Path, args):
    payload = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    if payload["tokenizer_hash"] != _file_hash(tok_path):
        raise ValueError("Tokenizer hash differs from checkpoint")
    if payload["configuration_hash"] != _config_hash(args):
        raise ValueError("Configuration differs from checkpoint")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["lr_scheduler"])
    curriculum.load_state_dict(payload["curriculum"])
    random.setstate(payload["python_rng"])
    rng.setstate(payload["local_python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_cpu_rng"])
    if payload["torch_cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng"])
    return payload


def overfit_test(tok, device, args):
    model, optimizer = make_model(tok, device, 731, args.layers, args.hidden,
                                  args.policy_lr)
    rng = random.Random(731); curriculum = Curriculum(False)
    fixed = list(range(1000)); history = []
    for rollout in range(args.overfit_rollouts):
        transitions, _, _ = collect_rollout(model, tok, device, curriculum, rng,
                                             min(512, len(fixed)), 0, "entity", fixed)
        metrics = ppo_update(model, tok, device, transitions, optimizer, rng,
                             args.target_kl)
        train_acc = accuracy(model, tok, device, "entity", (0, 1, 2, 3), 1000, fixed)
        history.append({"rollout": rollout + 1, "accuracy": train_acc, **metrics})
        print(f"overfit rollout={rollout+1} accuracy={train_acc:.3f} kl={metrics['kl']:.4f}", flush=True)
        if train_acc >= .95:
            return {"passed": True, "accuracy": train_acc, "history": history}
    return {"passed": False, "accuracy": history[-1]["accuracy"], "history": history}


def run_condition(condition, seed, budget, tok, tok_path, device, args,
                  run_dir: Path, resume_path: Path | None = None):
    policy_lr = 3e-4 if condition in ("iid_clm", "ordered_clm", "adaptive_clm", "adaptive_hybrid") else args.policy_lr
    model, optimizer = make_model(tok, device, seed, args.layers, args.hidden, policy_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    adaptive = condition in ("adaptive_caregiver_rl", "adaptive_hybrid", "adaptive_clm")
    curriculum = Curriculum(adaptive); rng = random.Random(seed + 19)
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "trace.json"
    checkpoint_path = run_dir / "latest.pt"
    trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {
        "condition": condition, "seed": seed, "config": _config_dict(args),
        "configuration_hash": _config_hash(args), "events": [], "termination_reason": "running",
    }
    visible = 0; rollout = 0
    if resume_path:
        payload = load_checkpoint(resume_path, model, optimizer, scheduler,
                                  curriculum, rng, tok_path, args)
        if payload["condition"] != condition or payload["seed"] != seed:
            raise ValueError("Resume condition or seed differs from checkpoint")
        visible, rollout = payload["visible_tokens"], payload["rollout"]
    next_diagnostic = ((visible // args.diagnostic_interval) + 1) * args.diagnostic_interval
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    while visible < budget and (args.max_rollouts is None or rollout < args.max_rollouts):
        progress = visible / budget
        step_started = time.perf_counter(); collection_metrics = {}
        if condition in ("iid_clm", "ordered_clm", "adaptive_clm"):
            situations = []
            for _ in range(64):
                skill = rng.choice(SKILLS) if condition == "iid_clm" else curriculum.sample(rng, progress)
                situations.append(make_situation(skill, rng.randrange(1 << 30), rng.randrange(4)))
            loss, used = clm_update(model, tok, device, situations, optimizer)
            visible += used; training_metrics = {"clm_loss": loss}
        else:
            transitions, used, collection_metrics = collect_rollout(
                model, tok, device, curriculum, rng, args.rollout_dialogues, progress)
            update_started = time.perf_counter()
            metrics = ppo_update(model, tok, device, transitions, optimizer, rng,
                                 args.target_kl)
            metrics["ppo_update_seconds"] = time.perf_counter() - update_started
            visible += used; training_metrics = metrics
            if condition == "adaptive_hybrid":
                env = [make_situation(rng.choice(SKILLS), rng.randrange(1 << 30), rng.randrange(4)) for _ in range(64)]
                # Environment-only text: no canonical child answer is included.
                texts = [Situation(s.skill, s.prompt + " " + s.explanation, [""], 0, "") for s in env]
                clm_loss, clm_tokens = clm_update(model, tok, device, texts, optimizer,
                                                  args.hybrid_clm_weight)
                visible += clm_tokens; training_metrics["environment_clm_loss"] = clm_loss
        scheduler.step(); rollout += 1
        diagnostic_metrics = None
        if adaptive and visible >= next_diagnostic:
            diagnostic_metrics = diagnostics(model, tok, device, args.diagnostic_items)
            curriculum.update_diagnostics(diagnostic_metrics, visible)
            next_diagnostic += args.diagnostic_interval
        elapsed = time.perf_counter() - step_started
        event = {
            "rollout": rollout, "visible_tokens": visible,
            "rollout_wall_seconds": elapsed,
            "tokens_per_second": used / max(elapsed, 1e-9),
            "training_metrics": training_metrics,
            "collection_metrics": collection_metrics,
            "diagnostic_metrics": diagnostic_metrics,
            "curriculum": curriculum.state_dict(),
            "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
            "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else 0,
        }
        trace["events"].append(event)
        trace["current_checkpoint"] = str(checkpoint_path)
        trace["visible_tokens"] = visible
        trace["wall_seconds_this_session"] = time.perf_counter() - started
        trace["termination_reason"] = "running"
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, visible,
                        rollout, curriculum, rng, condition, seed, tok_path, args)
        _atomic_json(trace_path, trace)
        print(f"  rollout={rollout} tokens={visible} tok/s={event['tokens_per_second']:.1f}", flush=True)
    completed = visible >= budget
    trace["termination_reason"] = "budget_complete" if completed else "max_rollouts_reached"
    _atomic_json(trace_path, trace)
    final = diagnostics(model, tok, device, args.diagnostic_items) if completed else None
    return {"condition": condition, "seed": seed, "visible_tokens": visible,
            "rollouts": rollout, "completed": completed,
            "wall_seconds_this_session": time.perf_counter() - started,
            "diagnostics": final, "curriculum": curriculum.state_dict(),
            "checkpoint": str(checkpoint_path), "trace": str(trace_path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=5_000_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[2000])
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(DEFAULT_CONDITIONS))
    p.add_argument("--rollout-dialogues", type=int, default=512)
    p.add_argument("--diagnostic-interval", type=int, default=100_000)
    p.add_argument("--diagnostic-items", type=int, default=128)
    p.add_argument("--overfit-test", action="store_true")
    p.add_argument("--overfit-rollouts", type=int, default=30)
    p.add_argument("--hybrid-clm-weight", type=float, choices=(.1, .3, 1.0), default=.3)
    p.add_argument("--policy-lr", type=float, default=1e-4,
                   help="Development-only PPO policy learning rate; freeze after calibration")
    p.add_argument("--target-kl", type=float, default=.03)
    p.add_argument("--max-rollouts", type=int)
    p.add_argument("--resume", type=Path)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--hidden", type=int, default=288)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    tokenizer_path = ROOT / "artifacts" / "stage0_tokenizer.model"
    build_tokenizer(tokenizer_path); tok = Tokenizer(tokenizer_path)
    device = torch.device(args.device); out = ROOT / "results"; out.mkdir(exist_ok=True)
    run_id = time.strftime("stage0-%Y%m%d-%H%M%S")
    if args.overfit_test:
        result = overfit_test(tok, device, args)
        path = out / f"{run_id}-overfit.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(path)
        raise SystemExit(0 if result["passed"] else 2)
    results = []
    if args.resume and (len(args.seeds) != 1 or len(args.conditions) != 1):
        p.error("--resume requires exactly one seed and one condition")
    for seed in args.seeds:
        for condition in args.conditions:
            print(f"running {condition} seed={seed}", flush=True)
            run_dir = args.resume.parent if args.resume else out / "runs" / f"{run_id}-{condition}-{seed}"
            results.append(run_condition(condition, seed, args.budget, tok,
                                         tokenizer_path, device, args, run_dir,
                                         args.resume))
            _atomic_json(out / f"{run_id}.json",
                         {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                          "results": results})


if __name__ == "__main__":
    main()
