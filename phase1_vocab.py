"""Phase 1: learn `code`, `math`, and `sentence` from caregiver-mediated textual experience."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from phase1_data import CONDITIONS, CONCEPTS
from phase1_runner import overfit_test, run

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=CONDITIONS, default="contingent_caregiver_rl")
    p.add_argument("--seed", type=int, default=5000); p.add_argument("--budget", type=int, default=2_000_000)
    p.add_argument("--device", default=""); p.add_argument("--tokenizer", default=str(ROOT / "artifacts" / "phase1_vocab.model"))
    p.add_argument("--caregiver-cache", default=str(ROOT / "artifacts" / "phase1_caregiver_cache.json"))
    p.add_argument("--vocab-size", type=int, default=2048); p.add_argument("--layers", type=int, default=8)
    p.add_argument("--hidden", type=int, default=288); p.add_argument("--heads", type=int, default=8); p.add_argument("--ff", type=int, default=1152)
    p.add_argument("--policy-lr", type=float, default=3e-5); p.add_argument("--clm-lr", type=float, default=3e-4)
    p.add_argument("--rollout-dialogues", type=int, default=256); p.add_argument("--clm-batch", type=int, default=64)
    p.add_argument("--ppo-minibatch", type=int, default=64); p.add_argument("--target-kl", type=float, default=.03)
    p.add_argument("--eval-interval", type=int, default=100_000); p.add_argument("--eval-items", type=int, default=128)
    p.add_argument("--output", default=str(ROOT / "results" / "phase1")); p.add_argument("--resume", default=None)
    p.add_argument("--max-steps", type=int, default=0); p.add_argument("--session-seconds", type=int, default=0)
    p.add_argument("--retention-tokens", type=int, default=200_000); p.add_argument("--withhold-concept", choices=CONCEPTS, default="code")
    p.add_argument("--overfit-rollouts", type=int, default=100); p.add_argument("--overfit-test", action="store_true")
    p.add_argument("--smoke-test", action="store_true"); return p.parse_args()


def smoke_test(args):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); args.budget = 3000; args.device = "cpu"; args.tokenizer = str(root / "tok.model")
        args.caregiver_cache = None; args.vocab_size = 512; args.layers = 1; args.hidden = 64; args.heads = 4; args.ff = 128
        args.policy_lr = 3e-4; args.clm_lr = 3e-4; args.rollout_dialogues = 8; args.clm_batch = 8; args.ppo_minibatch = 8
        args.target_kl = .2; args.eval_interval = 1500; args.eval_items = 8; args.output = str(root / "runs")
        args.max_steps = 4; args.session_seconds = 0; args.retention_tokens = 500; args.withhold_concept = "code"; args.resume = None
        result = run(args); assert result["steps"] == 4 and (Path(result["run_dir"]) / "latest.pt").exists()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        smoke_test(args); print("phase1 smoke test passed")
    elif args.overfit_test:
        result = overfit_test(args); print(json.dumps(result, indent=2)); raise SystemExit(0 if result["passed"] else 2)
    else:
        print(json.dumps(run(args), indent=2))
