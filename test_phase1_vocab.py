import random
import unittest
from pathlib import Path

from phase1_data import (
    CONCEPTS,
    RESPONSE_VOCAB,
    CaregiverLanguage,
    make_exemplar,
    make_lesson,
)
from phase1_rl import centered_reward, feedback


class Phase1VocabularyTests(unittest.TestCase):
    def setUp(self):
        cache_path = Path(__file__).parent / "artifacts" / "phase1_caregiver_cache.json"
        self.caregiver = CaregiverLanguage.load(str(cache_path))

    def test_training_and_transfer_families_are_structurally_distinct(self):
        for concept in CONCEPTS:
            train = {make_exemplar(concept, 17, f).text for f in range(4)}
            near = {make_exemplar(concept, 17, f).text for f in (4, 5)}
            far = {make_exemplar(concept, 17, f).text for f in (6, 7)}
            self.assertTrue(train.isdisjoint(near))
            self.assertTrue(train.isdisjoint(far))
            self.assertTrue(near.isdisjoint(far))

    def test_lesson_demonstrates_one_object_and_tests_another(self):
        lesson = make_lesson("code", 5, 0, self.caregiver)
        self.assertIn("code", lesson.demonstration)
        self.assertIn("Child:", lesson.prompt)
        self.assertNotIn(lesson.visible_text, lesson.demonstration)

    def test_shared_response_grammar_does_not_reveal_concept(self):
        self.assertEqual(set(RESPONSE_VOCAB).intersection(CONCEPTS), set(CONCEPTS))
        self.assertGreater(len(RESPONSE_VOCAB), len(CONCEPTS))

    def test_chance_centered_reward(self):
        n = len(RESPONSE_VOCAB)
        expected = (1.0 + (n - 1) * (-1.0 / (n - 1))) / n
        self.assertAlmostEqual(expected, 0.0)
        self.assertEqual(centered_reward(2, 2), 1.0)

    def test_yoked_feedback_teaches_an_unrelated_object(self):
        lesson = make_lesson("code", 9, 1, self.caregiver)
        text, retry, target = feedback("yoked_caregiver_rl", lesson, random.Random(3), self.caregiver)
        self.assertNotIn("this is code", text)
        self.assertEqual(target, "code")
        self.assertIn("what", retry.lower())

    def test_frozen_caregiver_cache_is_valid(self):
        rendered = self.caregiver.render("demonstration", 3, object="x = 1", label="code")
        self.assertIn("x = 1", rendered)
        self.assertIn("code", rendered)


if __name__ == "__main__":
    unittest.main()
