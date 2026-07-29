from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from phase1_data import (
    CONCEPTS,
    RESPONSE_VOCAB,
    CaregiverLanguage,
    Tokenizer,
    build_tokenizer,
    make_exemplar,
    make_lesson,
)
from phase1_runner import run


class Phase1DataTests(unittest.TestCase):
    def test_generator_families_are_separate(self):
        for concept in CONCEPTS:
            train = {make_exemplar(concept, seed, family).text for seed in range(20) for family in range(4)}
            hidden = {make_exemplar(concept, seed, family).text for seed in range(20) for family in range(4, 8)}
            self.assertFalse(train & hidden)

    def test_caregiver_cache_validation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cache.json"
            payload = {
                "metadata": {"source": "test"},
                "templates": {
                    "demonstration": ["{object} this is {label}"],
                    "question": ["{object} what is this"],
                    "correction": ["this is {label}"],
                    "analogy": ["{object} what is this"],
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            cache = CaregiverLanguage.load(str(path))
            self.assertEqual(cache.metadata["source"], "test")
            self.assertIn("code", cache.render("correction", 0, label="code", object="x=1"))

    def test_hidden_family_never_enters_lesson(self):
        caregiver = CaregiverLanguage.default()
        for concept in CONCEPTS:
            for seed in range(20):
                lesson = make_lesson(concept, seed, seed % 4, caregiver)
                self.assertNotIn(make_exemplar(concept, seed, 6).text, lesson.demonstration)

    def test_response_words_have_equal_token_length(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = Path(td) / "tok.model"
            build_tokenizer(model_path, CaregiverLanguage.default(), 512)
            tok = Tokenizer(model_path)
            lengths = {word: len(tok.encode(word)) for word in RESPONSE_VOCAB}
            self.assertEqual(len(set(lengths.values())), 1, lengths)


class Phase1RunnerTests(unittest.TestCase):
    def _args(self, root: Path, condition: str, max_steps: int = 2):
        return argparse.Namespace(
            condition=condition,
            seed=9100,
            budget=3000,
            device="cpu",
            tokenizer=str(root / "tok.model"),
            caregiver_cache=None,
            vocab_size=512,
            layers=1,
            hidden=64,
            heads=4,
            ff=128,
            policy_lr=3e-4,
            clm_lr=3e-4,
            rollout_dialogues=8,
            clm_batch=8,
            ppo_minibatch=8,
            target_kl=0.2,
            eval_interval=1500,
            eval_items=4,
            output=str(root / "runs"),
            resume=None,
            max_steps=max_steps,
            session_seconds=0,
            retention_tokens=500,
            withhold_concept="code",
            overfit_rollouts=3,
            overfit_test=False,
            smoke_test=False,
        )

    def test_all_conditions_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for condition in (
                "iid_clm",
                "ordered_clm",
                "trial_only_rl",
                "yoked_caregiver_rl",
                "contingent_caregiver_rl",
            ):
                args = self._args(root, condition)
                result = run(args)
                self.assertEqual(result["steps"], 2)
                self.assertTrue((Path(result["run_dir"]) / "result.json").exists())

    def test_resume_advances_same_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = self._args(root, "contingent_caregiver_rl", max_steps=1)
            first = run(args)
            checkpoint = Path(first["run_dir"]) / "latest.pt"
            args.resume = str(checkpoint)
            args.max_steps = 3
            second = run(args)
            self.assertEqual(first["run_dir"], second["run_dir"])
            self.assertEqual(second["steps"], 3)


if __name__ == "__main__":
    unittest.main()
