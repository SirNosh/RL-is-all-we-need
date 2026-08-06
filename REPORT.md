# Legacy contextual-bandit pilot report

Run date: 2026-07-26  
Result file: `results/20260726-193650.json`

## Correct verdict

This pilot did **not test the artificial-childhood hypothesis**.

It tested a narrower system: a randomly initialized 11.2M-parameter transformer
trained as a one-step, four-action contextual bandit with sparse binary reward.
That system did not outperform dense next-token training after 500,000 visible
tokens.

The artificial-childhood hypothesis remains untested because the pilot omitted
persistent dialogue, contingent correction, retries, demonstrations in context,
progressive textual production, mastery gating, genuine PPO, delayed transfer,
and retention.

## Held-out procedural accuracy

| Condition | Mean | SD across 3 seeds |
|---|---:|---:|
| IID CLM | 31.05% | 8.44% |
| Ordered CLM | 29.08% | 1.59% |
| Adaptive CLM | 26.65% | 4.42% |
| Fixed actor-critic | 25.06% | 1.40% |
| Adaptive actor-critic | 23.76% | 1.97% |
| Adaptive hybrid | 22.95% | 0.85% |

Chance is 25%. The paired adaptive-minus-IID differences were +1.74, -18.49,
and -5.12 percentage points. Three seeds are too imprecise for a scientific
contrast.

The old metric was called “far transfer,” but most templates were shared with
training. The accurate name is held-out procedural instances with partial
lexical changes.

## Concrete implementation defects

- Adaptive CLM never updated its scheduler from model diagnostics.
- The scheduler preferred later stages and used cumulative training reward.
- Ordered conditions had no spaced replay.
- The hybrid CLM signal was roughly 30 times weaker than standalone CLM.
- The tokenizer corpus included evaluation-side text.
- RL was a single actor-critic update, not stored-rollout clipped PPO.
- Responses were four special labels rather than language.

## Decision

The stop decision was correct: do not scale this implementation. The active
experiment is now the six-skill Stage-0 Textual Bootstrap Study. It must first
pass a 95% PPO overfit gate, then test caregiver-mediated communicative
bootstrap at five million visible tokens.
