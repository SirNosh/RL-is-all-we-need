# Phase 1: Text-Native Vocabulary Childhood

## Purpose

Phase 1 tests the first claim required by the larger artificial-childhood project:

> Can a randomly initialized autoregressive model acquire words by seeing textual objects, hearing a caregiver name them, attempting to use the names, receiving response-contingent teaching, and transferring the words to structurally new objects?

This replaces the previous Stage-0 task framing. The target is vocabulary grounding, not predefined question answering.

## Scope

The first lexicon contains five produced words:

- social routines: `hi`, `bye`;
- textual categories: `code`, `math`, `sentence`.

The primary scientific endpoint covers the three category words. Social words establish the earliest imitation routine but are not part of the primary macro score.

The child uses one shared response grammar:

```text
hi, bye, code, math, sentence, yes, no, again, unknown
```

The category is not revealed by a skill-specific action set. Open-ended token generation remains a later ablation and is not claimed here.

## What one developmental episode is

A label is demonstrated on one textual object:

```text
Caregiver: look.
print("hi")
Caregiver: this is code.
Caregiver: say code.
Child: code
```

The child is then tested on a different object from the same category:

```text
Caregiver: look at another one.
x = 4
Caregiver: what is this?
Child: math
```

In the contingent condition, the caregiver responds to that actual error and gives an analogous retry:

```text
Caregiver: not quite. this is code.
Caregiver: code.
Caregiver: try one more.
for i in range(3):
    print(i)
Caregiver: what is this?
Child: code
```

The object shown in the demonstration, first test, and retry is always different.

## Caregiver architecture

The caregiver has three separate parts:

1. A deterministic controller chooses the concept, verified exemplar, teaching action, retry, and reward.
2. A frozen LLM-authored cache supplies child-directed surface wording.
3. Deterministic verifiers know the category label and score the child.

`artifacts/phase1_caregiver_cache.json` was authored by a pretrained model and frozen before training. It contains no task examples. The caregiver model therefore controls how teaching is phrased, but it cannot change facts, curriculum allocation, rewards, or evaluation.

A live caregiver model is deliberately excluded from Phase 1 because it would give different children unequal and non-reproducible information. Live adaptive rendering is a later experiment after the mechanism is established.

## Textual object bank

Each category has eight independently implemented families:

- families 0–3: training;
- families 4–5: near transfer;
- families 6–7: far transfer.

### Code

Training covers assignment, printing, loops, and functions. Transfer changes the program structure to conditionals, list mutation, classes, and comprehensions.

### Mathematics

Training covers arithmetic, subtraction, comparison, and fractions. Transfer uses equations, area statements, symbolic functions, and geometry descriptions.

### Sentence

Training covers declaratives, questions, imperatives, and simple narratives. Transfer uses compound, passive, subordinate, and contrastive prose.

The tokenizer is trained only on training families and the frozen caregiver cache. Near and far exemplars never enter tokenizer construction.

## Conditions

| Condition | Child experience | Purpose |
|---|---|---|
| IID CLM | Gold caregiver-child transcripts shuffled across concepts | Dense static baseline |
| Ordered CLM | Gold transcripts in developmental order with 20% replay | Ordering without action or contingency |
| Trial-only RL | Child acts and receives scalar reward, with no teaching after errors | Tests reinforcement without demonstrations/corrections |
| Yoked caregiver RL | Same initial demonstrations and retry opportunities, but error feedback teaches an unrelated object | Controls feedback-token and demonstration budget without matched contingency |
| Contingent caregiver RL | Feedback names the actual object and retry remains in the same category | Proposed mechanism |

The critical contrast is:

```text
contingent caregiver RL - yoked caregiver RL
```

This isolates whether feedback being matched to the current child-object interaction matters beyond merely receiving more caregiver text.

## Student

The default child is the same random decoder architecture used in the completed Stage-0 study:

- 8 layers;
- hidden size 288;
- 8 attention heads;
- feed-forward size 1,152;
- RMSNorm, SwiGLU, RoPE, tied embeddings;
- 2,048-piece train-only BPE;
- scalar value head for PPO.

## Training

- Budget: 2,000,000 learner-visible tokens per run.
- CLM learning rate: `3e-4`.
- PPO learning rate: `3e-5`.
- PPO: two epochs, clip ratio 0.2, value coefficient 0.5, target-KL safeguard 0.03.
- Rollout: 256 initial dialogues.
- Wrong first attempts receive chance-centered negative reward.
- Correct first attempt: `+1.0`.
- Correct analogous retry: `+0.6`.
- Shared nine-response chance level: `11.11%`.
- Diagnostics every 100,000 visible tokens.
- Final 200,000 tokens withhold `code` while earlier vocabulary remains available for review.

All prompt, child-response, correction, demonstration, and retry tokens count toward the visible-token budget.

## Engineering gates

Before the development matrix:

1. `python phase1_vocab.py --overfit-test ...` must reach at least 95% on 1,000 fixed code-label situations using PPO.
2. Every condition must complete a 100,000-token mechanism run.
3. Checkpoint/resume must reproduce the uninterrupted trajectory under deterministic CPU execution.
4. No evaluation family may appear in the tokenizer corpus.
5. No NaN, OOM, invalid reward, or non-finite PPO statistic is permitted.

## Development matrix

Frozen development seeds: `5000`, `5001`, `5002`.

```text
5 conditions × 3 paired seeds × 2M tokens = approximately 30M visible tokens
```

These are development estimates, not confirmatory statistics.

## Evaluation

At each diagnostic and final endpoint, the model receives a new textual object and the question `what is this?` without a demonstration, correction, or retry.

For each category:

- 128 near-family items;
- 128 far-family items.

Primary endpoint:

```text
far-family macro accuracy over code, math, and sentence
```

Secondary endpoints:

- per-word neural age of acquisition, defined as the first two consecutive diagnostics above 70% far accuracy;
- near-to-far generalization gap;
- candidate-selection collapse;
- policy entropy;
- code retention before and after the final 200,000-token holdout;
- contingent-minus-yoked, contingent-minus-trial, and contingent-minus-IID paired differences.

## Development success gate

Proceed to a confirmatory study only if contingent caregiving satisfies all of the following in at least two of three seeds:

1. Far macro accuracy at least 70%.
2. Every category word at least 60% far accuracy.
3. At least a 10-point paired advantage over yoked caregiving.
4. At least a 10-point paired advantage over trial-only RL.
5. Code chance-adjusted retention ratio at least 0.80 when pre-holdout accuracy is above chance.
6. No scoring-rule or candidate-prior artifact explains the result.

## Interpretation

- Contingent > yoked > trial: matched caregiver feedback builds vocabulary.
- Yoked ≈ contingent > trial: demonstrations matter, but contingency does not.
- CLM > all RL conditions: dense prediction remains more efficient for initial lexical grounding.
- Training accuracy rises but far transfer stays low: the child memorizes surface features rather than category meaning.
- Only `code` succeeds: the experiment has found one separable textual category, not a general vocabulary mechanism.
- All conditions remain near chance: the current cold-start architecture or learning signal cannot establish the first lexicon.

## Scientific boundary

Passing Phase 1 would establish only that a random model can acquire three textual category words under a constrained shared response vocabulary. It would not establish open-ended language, broad childhood learning, mathematics, programming competence, or replacement of corpus pretraining.
