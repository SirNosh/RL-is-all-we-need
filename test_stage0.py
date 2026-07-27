import tempfile
import unittest
from pathlib import Path

import torch

from experiment import (PREREQUISITES, ROOT, SKILLS, Curriculum, Tokenizer,
                        _candidate_scores, make_model, make_situation)


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


if __name__ == "__main__":
    unittest.main()
