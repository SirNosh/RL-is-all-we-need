import unittest

import torch

from legacy_pilot import LABELS, SKILLS, Student, make_example


class ExperimentTests(unittest.TestCase):
    def test_all_36_generators_are_deterministic_and_valid(self):
        self.assertEqual(36, len(SKILLS))
        for skill, stage in SKILLS:
            a = make_example(skill, 123, "train")
            b = make_example(skill, 123, "train")
            self.assertEqual(a, b)
            self.assertIn(a.answer, LABELS)
            self.assertIn(a.answer, a.prompt)
            self.assertIn(stage, range(5))

    def test_far_split_changes_surface_form(self):
        for skill, _ in SKILLS:
            self.assertNotEqual(make_example(skill, 123, "train").prompt,
                                make_example(skill, 123, "far").prompt)

    def test_locked_model_size(self):
        model = Student(2048)
        count = sum(p.numel() for p in model.parameters())
        self.assertGreater(count, 11_000_000)
        self.assertLess(count, 13_000_000)
        logits, values = model(torch.randint(0, 2048, (2, 17)))
        self.assertEqual((2, 17, 2048), tuple(logits.shape))
        self.assertEqual((2, 17), tuple(values.shape))


if __name__ == "__main__":
    unittest.main()
