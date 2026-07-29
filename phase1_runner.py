from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from phase1_data import CONCEPTS, RESPONSE_VOCAB, CaregiverLanguage, Tokenizer, build_tokenizer, canonical_transcript, make_lesson, sample_lesson
from phase1_model import Student, candidate_scores, clm_loss
from phase1_rl import collect_rollout, ppo_update


def evaluate(model, tok, device, split, caregiver, items=128, seed=900000):
    families = {"near": (4, 5), "far": (6, 7)}[split]
    model.eval(); result = {}
    with torch.no_grad():
        for concept in CONCEPTS:
            lessons = [make_lesson(concept, seed + i, families[i % 2], caregiver) for i in range(items)]
            scores, _ = candidate_scores(model, tok, [x.prompt for x in lessons], RESPONSE_VOCAB, device)
            actions = scores.argmax(-1).cpu().tolist(); target = RESPONSE_VOCAB.index(concept)
            result[concept] = {"accuracy": sum(x == target for x in actions) / len(actions), "selections": dict(Counter(RESPONSE_VOCAB[x] for x in actions)), "mean_entropy": float(torch.distributions.Categorical(logits=scores).entropy().mean().cpu())}
    model.train(); result["macro_accuracy"] = float(np.mean([result[x]["accuracy"] for x in CONCEPTS])); return result


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8"); os.replace(tmp, path)


def run(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(args.seed)
    rng, caregiver = random.Random(args.seed), CaregiverLanguage.load(args.caregiver_cache)
    tokenizer_path = Path(args.tokenizer); build_tokenizer(tokenizer_path, caregiver, args.vocab_size); tok = Tokenizer(tokenizer_path)
    model = Student(tok.vocab_size, args.layers, args.hidden, args.heads, args.ff).to(device)
    lr = args.clm_lr if "clm" in args.condition else args.policy_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .95), weight_decay=.1)
    config = {
        "condition": args.condition, "seed": args.seed, "budget": args.budget, "layers": args.layers, "hidden": args.hidden,
        "heads": args.heads, "ff": args.ff, "policy_lr": args.policy_lr, "clm_lr": args.clm_lr,
        "rollout_dialogues": args.rollout_dialogues, "retention_tokens": args.retention_tokens, "withhold_concept": args.withhold_concept,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "caregiver_sha256": hashlib.sha256(Path(args.caregiver_cache).read_bytes()).hexdigest() if args.caregiver_cache else None,
        "caregiver_metadata": caregiver.metadata,
    }
    run_dir = Path(args.output) / f"phase1-{args.condition}-{args.seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    events, visible, step, retention_pre, holdout = [], 0, 0, None, False
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if payload["config"] != config: raise ValueError("resume configuration mismatch")
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        visible, step, events = payload["visible"], payload["step"], payload["events"]
        retention_pre, holdout = payload.get("retention_pre"), payload.get("holdout", False)
        rng.setstate(payload["rng"]); torch.set_rng_state(payload["torch_rng"])
        if device.type == "cuda" and payload.get("cuda_rng") is not None: torch.cuda.set_rng_state_all(payload["cuda_rng"])
    start, next_eval = time.perf_counter(), ((visible // args.eval_interval) + 1) * args.eval_interval
    while visible < args.budget:
        step += 1; progress = visible / args.budget
        if args.retention_tokens and not holdout and visible >= args.budget - args.retention_tokens:
            retention_pre = evaluate(model, tok, device, "far", caregiver, args.eval_items, 2_900_000); holdout = True
        withheld = frozenset((args.withhold_concept,)) if holdout else frozenset()
        if "clm" in args.condition:
            lessons = [sample_lesson(rng, progress, args.condition == "ordered_clm", caregiver, withheld) for _ in range(args.clm_batch)]
            texts = [canonical_transcript(x) for x in lessons]; loss = clm_loss(model, tok, texts, device)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            increment, metrics = sum(len(tok.encode(x)) + 1 for x in texts), {"clm_loss": float(loss.detach())}
        else:
            lessons = [sample_lesson(rng, progress, True, caregiver, withheld) for _ in range(args.rollout_dialogues)]
            transitions, increment, collection = collect_rollout(model, tok, args.condition, rng, lessons, caregiver, device)
            metrics = ppo_update(model, tok, optimizer, transitions, device, rng, minibatch=args.ppo_minibatch, target_kl=args.target_kl); metrics["collection"] = collection
        visible += increment; event = {"step": step, "visible_tokens": visible, "training": metrics, "elapsed_seconds": time.perf_counter() - start}
        if visible >= next_eval or visible >= args.budget:
            event["near"] = evaluate(model, tok, device, "near", caregiver, args.eval_items, 1_000_000 + step * 1000)
            event["far"] = evaluate(model, tok, device, "far", caregiver, args.eval_items, 2_000_000 + step * 1000); next_eval += args.eval_interval
        events.append(event)
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config, "visible": visible, "step": step, "events": events, "retention_pre": retention_pre, "holdout": holdout, "rng": rng.getstate(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        tmp = run_dir / "latest.pt.tmp"; torch.save(payload, tmp); os.replace(tmp, run_dir / "latest.pt")
        atomic_json(run_dir / "trace.json", {"config": config, "events": events})
        if args.max_steps and step >= args.max_steps: break
        if args.session_seconds and time.perf_counter() - start >= args.session_seconds: break
    far = evaluate(model, tok, device, "far", caregiver, args.eval_items, 4_000_000); retention = None
    if retention_pre is not None:
        chance = 1 / len(RESPONSE_VOCAB); pre, post = retention_pre[args.withhold_concept]["accuracy"], far[args.withhold_concept]["accuracy"]
        retention = {"concept": args.withhold_concept, "pre": pre, "post": post, "drop": pre-post, "chance_adjusted_ratio": None if pre <= chance else (post-chance)/(pre-chance)}
    final = {"config": config, "visible_tokens": visible, "steps": step, "complete": visible >= args.budget, "near": evaluate(model, tok, device, "near", caregiver, args.eval_items, 3_000_000), "far": far, "retention": retention, "run_dir": str(run_dir)}
    atomic_json(run_dir / "result.json", final); return final


def overfit_test(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng, caregiver = random.Random(args.seed), CaregiverLanguage.load(args.caregiver_cache)
    path = Path(args.tokenizer); build_tokenizer(path, caregiver, args.vocab_size); tok = Tokenizer(path)
    model = Student(tok.vocab_size, args.layers, args.hidden, args.heads, args.ff).to(device); opt = torch.optim.AdamW(model.parameters(), lr=args.policy_lr)
    fixed = [make_lesson("code", i, i % 4, caregiver) for i in range(1000)]; history = []
    for rollout in range(1, args.overfit_rollouts + 1):
        batch = [fixed[rng.randrange(1000)] for _ in range(args.rollout_dialogues)]
        transitions, _, collection = collect_rollout(model, tok, "contingent_caregiver_rl", rng, batch, caregiver, device)
        update = ppo_update(model, tok, opt, transitions, device, rng, minibatch=args.ppo_minibatch, target_kl=args.target_kl)
        if rollout == 1 or rollout % 5 == 0:
            with torch.no_grad():
                scores, _ = candidate_scores(model, tok, [x.demonstration + "\n" + x.prompt for x in fixed], RESPONSE_VOCAB, device)
                accuracy = float((scores.argmax(-1) == RESPONSE_VOCAB.index("code")).float().mean().cpu())
            history.append({"rollout": rollout, "accuracy": accuracy, "collection": collection, "update": update})
            if accuracy >= .95: return {"passed": True, "accuracy": accuracy, "rollouts": rollout, "history": history}
    return {"passed": False, "accuracy": history[-1]["accuracy"], "rollouts": args.overfit_rollouts, "history": history}
