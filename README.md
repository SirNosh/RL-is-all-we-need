# Stage-0 Textual Bootstrap Study

The active experiment asks whether a randomly initialized text model can learn
six elementary communicative conventions from contingent textual caregiving.

The earlier 36-skill four-choice run is preserved in `legacy_pilot.py`,
`REPORT.md`, and `results/20260726-193650.json`. It did not instantiate
artificial childhood and is not evidence against that hypothesis.

The Stage-0 study covers:

- turn-taking, yes/no judgment, entity reference, property binding, counting
  0–5, and requesting missing information;
- short textual replies rather than `<A>`–`<D>` actions;
- multi-turn correction, demonstrations, retries, and spaced review;
- tokenizer construction from training text only;
- hidden diagnostics and independently authored transfer families;
- clipped PPO using frozen rollout probabilities and two optimization epochs.

`adaptive_clm` is available as an additional diagnostic-driven development
control, but is not one of the five default confirmatory conditions.

## Setup

```powershell
python -m pip install -r requirements.txt
python -m unittest -v
```

The reported legacy run used Python 3.12, PyTorch 2.6.0+cu124, NumPy 2.3.1,
SentencePiece 0.2.0, and an RTX 4070 Super. NVIDIA users should install the
PyTorch wheel appropriate to their CUDA runtime.

## Mandatory optimizer gate

Before scientific runs, PPO must reach 95% training accuracy on one skill and
1,000 fixed situations:

```powershell
python experiment.py --overfit-test
```

## Stage-0 matrix

The bounded 500k mechanism gate passed on engineering seed 3201. Run one
seed-condition pair per command, with graceful session limits:

```powershell
python experiment.py --budget 5000000 --seeds 4000 --conditions iid_clm --session-seconds 1000
```

Runs save `results/runs/<run>/latest.pt` and atomically update `trace.json`
after every rollout. A session-limit exit is clean and resumable:

```powershell
python experiment.py --budget 5000000 --seeds 4000 --conditions iid_clm --session-seconds 1000 --resume results/runs/<run>/latest.pt
```

Resume identity is based on the seed-condition scientific configuration,
tokenizer, executable experiment/model files, and pinned requirements. Outer
matrix lists, output paths, session controls, and documentation-only commits do
not invalidate a checkpoint.

Seeds 1000–1002 were inspected during the legacy pilot and are excluded from
the new confirmatory study. Development also excludes overfit seed 731,
interrupted seed 3000, calibration seeds 3100–3102, and engineering seeds
3200–3201.

For 5M-token runs, hidden far-family performance on `turn` and `entity` is
recorded at the first batch/rollout boundary after 4.5M tokens. A dedicated
`retention_start.pt` is saved, the two skills are withheld, and post-holdout
performance is evaluated on different seeds. Traces also record raw and
length-normalized accuracy, candidate token lengths and selections, and policy
entropy.

All three corrected IID CLM development seeds have completed. These summarize
one condition, not a cross-condition comparison. See `STAGE0_REPORT.md` and
the raw result JSON files under `results/`.

All 15 endpoints are complete: three seeds each for IID CLM, ordered CLM,
fixed caregiver RL, adaptive caregiver RL, and adaptive hybrid. The exact
review paths are listed in `results/development_v2_manifest.json`.

Adaptive caregiver primary far performance averaged 83.3% turn, 65.9% truth,
19.3% entity, 46.6% property, 29.4% counting, and 0% clarification. Relative
to paired IID seeds, those differences were +66.7, 0, âˆ’1.3, âˆ’9.1, +4.2, and
âˆ’46.6 percentage points. Length normalization reduces adaptive turn to 66.7%
but leaves a +66.7-point paired advantage over IID.

Adaptive scheduling therefore retains a large interaction-specific turn
advantage over IID, but it is less seed-robust than fixed caregiving: adaptive
minus fixed is âˆ’16.7 points primary and âˆ’33.3 points length-normalized for
turn. It does not improve the broader suite.

The complete hybrid condition averages 66.7% turn, 72.4% truth, 15.4% entity,
65.1% property, 24.7% counting, and 16.7% clarification under primary scoring.
Its turn and property estimates have large seed variance. Length normalization
changes hybrid turn to 33.3% and clarification to 100%, exposing a major
conventional-reply scoring dependency.

The Stage-0 result is narrow but positive: fixed contingent caregiving teaches
and retains turn-taking robustly, while IID and ordered CLM do not. Adaptive
and hybrid scheduling weaken that robustness, and no interactive condition
produces a broad advantage across the other five skills. These are
three-development-seed estimates, not confirmatory statistics.
