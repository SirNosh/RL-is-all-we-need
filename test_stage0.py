import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from experiment import (PREREQUISITES, ROOT, SKILLS, Curriculum, Tokenizer,
                        _candidate_scores, make_model, make_situation,
                        run_condition)


class StageZeroTests(unittest.TestCase):
    def test_eight_families_and_answers(self):
        for skill in SKILLS:
            prompts = []
            for family in range(8):
                situation = make_situation(skill, 42, family)
                self.assertIn(situation.answer, range(len(situation.candidates)))
                self.assertLessEqual(len(situation.candidates[situation.answer].split()), 8)
                prompts.append(situation.prompt)
            self.assertEqual(8, len(set(prompts)))

    def test_prerequisites_gate_unlocking(self):
        curriculum = Curriculum(True)
        self.assertEqual(["turn"], curriculum.eligible())
        passing = {s: {"independent": 0, "far": 0} for s in SKILLS}
        passing["turn"] = {"independent": .9, "far": .8}
        curriculum.update_diagnostics(passing, 100_000)
        self.assertNotIn("turn", curriculum.mastered)
        curriculum.update_diagnostics(passing, 200_000)
        self.assertIn("turn", curriculum.mastered)
        self.assertIn("truth", curriculum.eligible())
        self.assertIn("entity", curriculum.eligible())
        self.assertNotIn("property", curriculum.eligible())

    def test_frontier_probes_do_not_unlock_skills(self):
        curriculum = Curriculum(True)
        rng = __import__("random").Random(12)
        for _ in range(2000):
            curriculum.sample(rng, 0)
        self.assertGreater(curriculum.frontier_probes, 0)
        self.assertEqual(set(), curriculum.mastered)
        self.assertEqual(["turn"], curriculum.eligible())
        self.assertEqual(set(SKILLS), set(curriculum.skill_selections))

    def test_evaluation_names_absent_from_training_families(self):
        evaluation_names = {"Asha", "Bram", "Inez", "Keon", "Yuki", "Zuri"}
        training = " ".join(make_situation(skill, seed, family).prompt
                            for skill in SKILLS for seed in range(20) for family in range(4))
        self.assertTrue(evaluation_names.isdisjoint(training.split()))

    def test_text_candidate_policy_shapes(self):
        tok = Tokenizer(ROOT / "artifacts" / "stage0_tokenizer.model")
        self.assertEqual(2048, tok.sp.vocab_size())
        model, _ = make_model(tok, torch.device("cpu"), 9, layers=2, hidden=96)
        situation = make_situation("truth", 4, 0)
        scores, values = _candidate_scores(
            model, tok, ["Caregiver: " + situation.prompt + "\nChild:"],
            [situation.candidates], torch.device("cpu"))
        self.assertEqual((1, 2), tuple(scores.shape))
        self.assertEqual((1,), tuple(values.shape))

    def test_rollout_boundary_resume_is_exact_on_cpu(self):
        tok_path = ROOT / "artifacts" / "stage0_tokenizer.model"
        tok = Tokenizer(tok_path)
        base = dict(
            budget=10**9, seeds=[41], conditions=["fixed_caregiver_rl"],
            rollout_dialogues=8, diagnostic_interval=100_000,
            diagnostic_items=2, overfit_test=False, overfit_rollouts=1,
            hybrid_clm_weight=.3, layers=1, hidden=32, device="cpu",
            policy_lr=1e-4, target_kl=.03, resume=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            continuous_args = Namespace(**base, max_rollouts=2)
            run_condition("fixed_caregiver_rl", 41, 10**9, tok, tok_path,
                          torch.device("cpu"), continuous_args, root / "continuous")
            split_args = Namespace(**base, max_rollouts=1)
            run_condition("fixed_caregiver_rl", 41, 10**9, tok, tok_path,
                          torch.device("cpu"), split_args, root / "split")
            resume_args = Namespace(**base, max_rollouts=2)
            run_condition("fixed_caregiver_rl", 41, 10**9, tok, tok_path,
                          torch.device("cpu"), resume_args, root / "split",
                          root / "split" / "latest.pt")
            continuous = torch.load(root / "continuous" / "latest.pt",
                                    map_location="cpu", weights_only=False)
            resumed = torch.load(root / "split" / "latest.pt",
                                 map_location="cpu", weights_only=False)
            for name, value in continuous["model"].items():
                self.assertTrue(torch.equal(value, resumed["model"][name]), name)
            self.assertEqual(continuous["curriculum"], resumed["curriculum"])
            self.assertEqual(continuous["visible_tokens"], resumed["visible_tokens"])
            self.assertEqual(continuous["rollout"], resumed["rollout"])


if __name__ == "__main__":
    unittest.main()
