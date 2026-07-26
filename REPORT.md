# Experiment report

Run date: 2026-07-26  
Result file: `results/20260726-193650.json`

## Outcome

The 500,000-visible-token engineering pilot did **not** support an adaptive-RL
advantage. With four answer choices, chance accuracy is 25%. Adaptive
interactive RL averaged 23.76% far-transfer accuracy; IID CLM averaged 31.05%.
The paired adaptive-RL minus IID-CLM difference was -7.29 percentage points.

This is a pilot result, not the locked confirmatory experiment. It cannot
establish that the proposed childhood paradigm fails. It establishes that this
implementation and exposure budget do not yet qualify for the protocol's
five-million-token engineering gate and should not be scaled unchanged.

## Far-transfer macro accuracy

| Condition | Mean | SD across 3 seeds |
|---|---:|---:|
| IID CLM | 31.05% | 8.44% |
| Ordered CLM | 29.08% | 1.59% |
| Adaptive CLM | 26.65% | 4.42% |
| Fixed interactive RL | 25.06% | 1.40% |
| Adaptive interactive RL | 23.76% | 1.97% |
| Adaptive hybrid | 22.95% | 0.85% |

The paired adaptive-RL minus IID-CLM differences for seeds 1000, 1001, and
1002 were +1.74, -18.49, and -5.12 percentage points.

## Engineering-gate check

The protocol requires adaptive RL to reach L1 80%, L3 75%, L4 70%, and M1 65%
within five million visible tokens in at least two of three development seeds.
At 0.5 million tokens, the three-seed means were:

| Skill | Observed | Required |
|---|---:|---:|
| L1 | 24.0% | 80% |
| L3 | 20.8% | 75% |
| L4 | 26.0% | 70% |
| M1 | 17.7% | 65% |

Because this check is at 0.5 million rather than the prescribed 5 million
tokens, it is not a formal gate failure. It does provide no basis for proceeding
directly to the 60 locked confirmatory runs.

## Scope and integrity

- Architecture: 11,211,841 parameters, 8 layers, hidden size 288, 8 heads,
  SwiGLU, RMSNorm, RoPE, tied embeddings, 2,048-piece BPE tokenizer.
- Curriculum: all 36 specified competency IDs across five stages.
- Evaluation: 32 deterministic far-transfer items per skill, 1,152 items per
  model, greedy one-choice decoding.
- Runs: six conditions × three paired initializations.
- Exposure: at least 500,000 learner-visible tokens per run.
- Total measured training/evaluation wall time: 498 seconds.
- Peak allocated GPU memory: 625 MB.

The implementation is narrower than the locked protocol: responses are
single-token multiple choice; interactive conditions currently use
deterministic verifier reward but not the full three-attempt contingent
caregiver loop; the policy update is a one-batch actor-critic update rather than
the fully specified two-epoch clipped PPO; retention, capstones,
learning-to-learn, and confirmatory statistics are absent. Seeds 1000–1002 have
now been inspected during development and must not later be represented as
untouched confirmatory seeds if the implementation is revised.

## Decision

Do not spend the specified 3 billion training tokens on this version. The
smallest useful next experiment is to make Stage 0 bootstrap reliably under the
prescribed caregiver retries and clipped PPO, validate it with separate
development seeds at five million tokens, and only then freeze the system.
