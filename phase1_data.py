from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm

CONCEPTS = ("code", "math", "sentence")
SOCIAL = ("hi", "bye")
RESPONSE_VOCAB = ("hi", "bye", "code", "math", "sentence", "yes", "no", "again", "unknown")
CONDITIONS = ("iid_clm", "ordered_clm", "trial_only_rl", "yoked_caregiver_rl", "contingent_caregiver_rl")


@dataclass(frozen=True)
class Exemplar:
    concept: str
    text: str
    family: int
    seed: int


@dataclass(frozen=True)
class Lesson:
    concept: str
    prompt: str
    target: str
    demonstration: str
    correction: str
    analogous_prompt: str
    analogous_target: str
    visible_text: str


@dataclass(frozen=True)
class CaregiverLanguage:
    metadata: dict
    templates: dict[str, tuple[str, ...]]

    @staticmethod
    def default() -> "CaregiverLanguage":
        return CaregiverLanguage(
            {"source": "built-in deterministic control"},
            {
                "demonstration": ("Caregiver: look.\n{object}\nCaregiver: this is {label}.\nCaregiver: say {label}.\nChild: {label}",),
                "question": ("Caregiver: look at another one.\n{object}\nCaregiver: what is this?\nChild:",),
                "correction": ("Caregiver: not quite. this is {label}.\nCaregiver: {label}.",),
                "analogy": ("Caregiver: look again.\n{object}\nCaregiver: what is this?\nChild:",),
            },
        )

    @staticmethod
    def load(path: str | None) -> "CaregiverLanguage":
        if not path:
            return CaregiverLanguage.default()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {"demonstration", "question", "correction", "analogy"}
        groups = payload.get("templates", {})
        missing = required - set(groups)
        if missing:
            raise ValueError(f"caregiver cache missing {sorted(missing)}")
        normalized = {}
        for key in required:
            values = tuple(map(str, groups[key]))
            if not values:
                raise ValueError(f"empty caregiver group: {key}")
            for value in values:
                if key in {"demonstration", "question", "analogy"} and "{object}" not in value:
                    raise ValueError(f"{key} must contain {{object}}")
                if key in {"demonstration", "correction"} and "{label}" not in value:
                    raise ValueError(f"{key} must contain {{label}}")
            normalized[key] = values
        return CaregiverLanguage(dict(payload.get("metadata", {})), normalized)

    def render(self, key: str, seed: int, **values: str) -> str:
        choices = self.templates[key]
        return choices[seed % len(choices)].format(**values)


def _rng(concept: str, seed: int, family: int) -> random.Random:
    return random.Random(seed * 1_000_003 + family * 97_409 + sum(map(ord, concept)) * 1_009)


def make_exemplar(concept: str, seed: int, family: int) -> Exemplar:
    if concept not in CONCEPTS or not 0 <= family < 8:
        raise ValueError("invalid concept or family")
    r = _rng(concept, seed, family)
    if concept == "code":
        name, n = r.choice(("x", "total", "count", "value", "item", "result")), r.randint(1, 9)
        examples = (
            f"{name} = {n}", f'print("{r.choice(("hi", "ready", "done"))}")',
            f"for {name} in range({n}):\n    print({name})", f"def add_{name}(a, b):\n    return a + b",
            f"if {name} > {n}:\n    {name} = {name} - 1", f"items = []\nitems.append({n})",
            f"class Counter:\n    def step(self):\n        self.value += {n}", f"squares = [k * k for k in range({n})]",
        )
    elif concept == "math":
        a, b = r.randint(1, 12), r.randint(1, 12)
        examples = (
            f"{a} + {b} = {a+b}", f"{a+b} - {a} = {b}", f"{a} < {a+b}", f"1/{a} + 1/{a} = 2/{a}",
            f"Solve: q + {a} = {a+b}", f"The area is {a} × {b} = {a*b} square units.",
            f"f(t) = t² + {a}t + {b}", "A triangle has angles 90°, 45°, and 45°.",
        )
    else:
        animal, place = r.choice(("dog", "cat", "bird", "fox")), r.choice(("garden", "river", "room", "hill"))
        examples = (
            f"The {animal} ran to the {place}.", f"Is the {animal} near the {place}?",
            f"Please open the door before the {animal} arrives.", f"Yesterday, the {animal} rested beside the {place}.",
            f"Although it was late, the {animal} quietly crossed the {place}.",
            f"The {animal} was followed by a friend who knew the way home.",
            f"After the rain stopped, everyone returned to the {place}; the {animal} stayed outside.",
            f"What surprised the traveler was not the {place}, but how calmly the {animal} waited.",
        )
    return Exemplar(concept, examples[family], family, seed)


def make_social_lesson(seed: int, greeting: bool) -> Lesson:
    if greeting:
        utterance = random.Random(seed * 101).choice(("hi", "hello"))
        return Lesson("hi", f"Caregiver: {utterance}\nChild:", "hi", "Caregiver: hi\nChild: hi", "Caregiver: say hi.", "Caregiver: hello\nChild:", "hi", utterance)
    return Lesson("bye", "Caregiver: we are finished. bye.\nChild:", "bye", "Caregiver: bye\nChild: bye", "Caregiver: say bye.", "Caregiver: goodbye.\nChild:", "bye", "bye")


def make_lesson(concept: str, seed: int, family: int, caregiver: CaregiverLanguage) -> Lesson:
    test = make_exemplar(concept, seed, family)
    demo = make_exemplar(concept, seed + 3571, (family + 1) % 4)
    analog = make_exemplar(concept, seed + 7919, (family + 2) % 4)
    return Lesson(
        concept,
        caregiver.render("question", seed + family * 13, object=test.text, label=concept),
        concept,
        caregiver.render("demonstration", seed + family * 11, object=demo.text, label=concept),
        caregiver.render("correction", seed + family * 17, object=test.text, label=concept),
        caregiver.render("analogy", seed + family * 19, object=analog.text, label=concept),
        concept,
        test.text,
    )


def ordered_sequence(progress: float) -> tuple[str, tuple[str, ...]]:
    seq, bounds = ("hi", "bye", "code", "math", "sentence"), (0.05, 0.10, 0.40, 0.70, 1.01)
    i = next(i for i, boundary in enumerate(bounds) if progress < boundary)
    return seq[i], seq[:i]


def sample_lesson(rng: random.Random, progress: float, ordered: bool, caregiver: CaregiverLanguage, withheld=frozenset()) -> Lesson:
    if ordered:
        current, previous = ordered_sequence(progress)
        review = [x for x in previous if x not in withheld]
        if review and rng.random() < 0.20:
            concept = rng.choice(review)
        else:
            available = [x for x in (current,) + previous if x not in withheld]
            concept = available[0] if available else "sentence"
    else:
        population = [x for x in ("hi", "bye") + CONCEPTS if x not in withheld]
        weights = {"hi": .05, "bye": .05, "code": .30, "math": .30, "sentence": .30}
        concept = rng.choices(population, weights=[weights[x] for x in population], k=1)[0]
    if concept in SOCIAL:
        return make_social_lesson(rng.randrange(1_000_000), concept == "hi")
    return make_lesson(concept, rng.randrange(1_000_000), rng.randrange(4), caregiver)


def canonical_transcript(lesson: Lesson) -> str:
    return lesson.demonstration + "\n" + lesson.prompt + " " + lesson.target


class Tokenizer:
    def __init__(self, path: Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(path))
        self.pad, self.eos, self.vocab_size = self.sp.pad_id(), self.sp.eos_id(), self.sp.vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)


def build_tokenizer(path: Path, caregiver: CaregiverLanguage, vocab_size: int = 2048) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    corpus = path.with_suffix(".corpus.txt")
    with corpus.open("w", encoding="utf-8") as f:
        for seed in range(2500):
            for concept in CONCEPTS:
                for family in range(4):
                    lesson = make_lesson(concept, seed, family, caregiver)
                    f.write(canonical_transcript(lesson) + "\n" + lesson.correction + "\n" + lesson.analogous_prompt + " " + concept + "\n")
            f.write(make_social_lesson(seed, True).demonstration + "\n" + make_social_lesson(seed, False).demonstration + "\n")
        for response in RESPONSE_VOCAB:
            f.write(response + "\n")
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=*/()[]{}:;,.!?<>\"'"
        for a in alphabet:
            for b in alphabet:
                f.write(f"reserve {a}{b}\n")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(path.with_suffix("")), model_type="bpe", vocab_size=vocab_size, byte_fallback=True, pad_id=0, unk_id=1, bos_id=2, eos_id=3, character_coverage=1.0)
