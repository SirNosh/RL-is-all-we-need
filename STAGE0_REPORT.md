# Stage-0 engineering report

Date: 2026-07-26  
Status: mechanism gates passed; corrected development matrix pending

## PPO overfit gate

The prescribed precondition was to reach at least 95% training accuracy on one
skill and 1,000 fixed situations. The Stage-0 entity-reference policy reached
98.5% after seven rollouts of 512 caregiver dialogues.

| Rollout | Training accuracy |
|---:|---:|
| 1 | 20.7% |
| 2 | 23.7% |
| 3 | 35.2% |
| 4 | 58.3% |
| 5 | 78.5% |
| 6 | 94.4% |
| 7 | 98.5% |

This is an optimizer/reward validation, not evidence for transfer or artificial
childhood. The policy uses an explicit skill-appropriate textual response
grammar to keep cold-start exploration finite. It emits real response strings,
but unconstrained vocabulary generation remains a future ablation.

Approximate KL reached 0.202 and the last clip fraction was 0.536. Those values
show aggressive policy movement despite clipping and must be monitored during
calibration. They do not invalidate the overfit result, but they rule out
calling the optimizer fully tuned.

## Bounded development run

A fresh-seed 500,000-token adaptive-caregiver run was started only to validate
the end-to-end scheduler. It was stopped without a result after approximately
13 minutes because it used about 11.8 GB of the shared RTX 4070 Super and other
projects also require local resources.

No scientific endpoint is reported from the interrupted run.

Future GPU work is coordinated through
`C:\Users\devya\OneDrive\Desktop\resources.txt` under the name
`rl-is-all-we-need`. No experiment may start without an active reservation and
an estimated release time in that ledger.

## Resume and resource engineering

- Continuous versus split CPU execution is bit-identical at rollout boundaries.
- CUDA checkpoint resume was exercised successfully after fixing CPU RNG state
  deserialization.
- Every rollout now atomically writes a 128 MB latest checkpoint and a JSON
  trace containing tokens, timings, PPO metrics, diagnostics, curriculum state,
  dialogue-ID hash, and allocated/reserved CUDA memory.
- Reducing candidate-scoring chunks from 64 to 32 cut calibration peak
  allocation from approximately 11.8 GB to 5.7 GB.

At the original PPO learning rate, calibration showed rollout-mean KL above
0.4 and more than 85% clipping. The development setting was changed to a
`3e-5` policy learning rate with a target-KL safeguard at 0.03. The overfit
gate re-passed at 95.5%, now requiring 68 rollouts. The stabilized 81k-token
calibration used 5.63 GB allocated VRAM and completed without numerical error.

## First 500k mechanism gate

Engineering seed 3200 completed 516,170 tokens across 32 rollouts, including a
deliberate stop and successful CUDA resume. Five diagnostics ran, retries
occurred, sampling changed after `turn` mastery, and there were no NaNs.

The gate nevertheless failed mechanism coverage: strict prerequisites limited
training to `turn`, `truth`, and `entity`; property binding, counting, and
clarification received no training. This is not a scientific result.

The next development revision retains mastery-gated priority while assigning 5%
of interactions to explicitly logged frontier probes among locked skills. These
probes provide mechanism coverage without marking a skill unlocked or mastered.

## Repeated 500k mechanism gate

Engineering seed 3201 passed the mechanism-operation gate:

- 506,795 visible tokens and 30 rollout-boundary checkpoints;
- five hidden diagnostic checkpoints;
- all six skills received exposure, including 759 logged frontier probes;
- 5,282 caregiver retries;
- 19.4% review allocation when review was available;
- attempt-1 training accuracy rose from 57.0% to 78.1%;
- no NaNs or CUDA failures;
- sampling changed after `turn` met the two-checkpoint mastery rule;
- final evaluation used hidden generator families.

Only `turn` mastered. Final far-family accuracy was 100% turn-taking, 53.1%
truth judgment, 18.8% entity reference, 37.5% property binding, 29.7% counting,
and 0% clarification. These figures are engineering observations, not a
condition comparison or hypothesis result.

The largest rollout-mean approximate KL was 0.0906; target-KL early stopping
prevented continued updates from that rollout. This remains a monitoring item.

## Frozen development scope

The three development seeds are `4000`, `4001`, and `4002`. Excluded seeds are
1000–1002 (legacy), 731 (overfit), 3000 (interrupted run), 3100–3102
(calibration), and 3200–3201 (engineering gates).

Each 5M-token run withholds `turn` and `entity` during the final 500,000 tokens
for a prespecified hidden-family retention check. The five development
conditions remain IID CLM, ordered CLM with replay, fixed caregiver RL,
adaptive caregiver RL, and adaptive hybrid.

## Development study progress

IID CLM seed 4000 produced the first completed 5M-token development-v1
endpoint:

| Skill | Independent | Far family |
|---|---:|---:|
| Turn-taking | 0.0% | 0.0% |
| Truth judgment | 71.9% | 78.1% |
| Entity reference | 14.1% | 18.0% |
| Property binding | 50.0% | 62.5% |
| Counting | 47.7% | 21.1% |
| Clarification | 100.0% | 100.0% |

After a 500,756-token interval without direct turn-taking or entity-reference
examples, final far-family accuracy was 0% and 18.0%, respectively. Because v1
did not evaluate immediately before withholding, these are post-holdout scores,
not measured forgetting. This is one development seed, not a condition
comparison.

The v1 endpoint is preserved as `development_v1_pre_retention_baseline`. Seed
4001 stopped at 36,830 tokens and is preserved as
`interrupted_unresumable_under_v1`: its checkpoint hashes the outer matrix
lists, requires the old Git SHA, and predates scientific-file hashes. Seed 4002
did not start.

The corrected protocol uses per-run resume identity, executable-file hashes, a
pre-holdout baseline and dedicated checkpoint, distinct post-holdout seeds,
candidate-length diagnostics, and graceful rollout-boundary session exits.
Development seeds 4000–4002 restart from scratch under corrected frozen commit
`87c5773`.

## Corrected IID CLM seed 4000

The corrected run completed 5,000,756 visible tokens across 271 logical
rollouts. A 1,000-second session stopped cleanly at 4,227,010 tokens and resumed
to the identical run directory. The retention baseline was recorded at
4,501,006 tokens, with 499,750 subsequent tokens withholding `turn` and
`entity`.

| Skill | Independent | Far | Length-normalized far |
|---|---:|---:|---:|
| Turn-taking | 0.0% | 0.0% | 0.0% |
| Truth judgment | 71.9% | 78.1% | 78.1% |
| Entity reference | 14.1% | 18.0% | 18.8% |
| Property binding | 50.0% | 62.5% | 62.5% |
| Counting | 47.7% | 21.1% | 21.1% |
| Clarification | 100.0% | 100.0% | 100.0% |

| Retention skill | Pre-holdout far | Post-holdout far | Pre − post |
|---|---:|---:|---:|
| Turn-taking | 0.0% | 0.0% | 0.0 pp |
| Entity reference | 9.4% | 16.4% | −7.0 pp |

Chance is 25% for turn-taking and 16.7% for entity reference. Neither
pre-holdout score was meaningfully above chance, so no retention ratio is
reported and this run provides no evidence of forgetting: the two held-out
skills had not been acquired at the retention boundary.

Length normalization leaves the endpoint almost unchanged. The extreme
turn/clarification scores are also not explained by a universal short-response
preference: the final policy selected the longer canonical clarification reply
on all 128 far items, while selecting the wrong `please tell me` reply on all
turn items. Candidate selection and entropy details are preserved in the raw
trace.

The accurate single-seed conclusion is heterogeneous acquisition. IID CLM
learned truth judgment, property binding, and the fixed clarification
convention; counting transferred poorly from independent to far families;
turn-taking and entity reference remained weak. No comparison among training
conditions exists.

## Corrected IID CLM seed 4001

Seed 4001 completed 5,002,499 visible tokens across 271 logical rollouts,
including one clean session-limit resume. Its retention baseline was recorded
at 4,501,471 tokens.

| Skill | Independent | Far | Length-normalized far |
|---|---:|---:|---:|
| Turn-taking | 100.0% | 50.0% | 0.0% |
| Truth judgment | 85.2% | 66.4% | 66.4% |
| Entity reference | 14.1% | 21.9% | 18.8% |
| Property binding | 32.0% | 57.0% | 57.0% |
| Counting | 71.1% | 40.6% | 40.6% |
| Clarification | 50.0% | 39.8% | 100.0% |

Primary retention scores changed from 0% to 50% for turn-taking and from 17.2%
to 22.7% for entity reference. The corresponding length-normalized scores
changed from 0% to 0% and from 16.4% to 17.2%. Neither pre-holdout score was
meaningfully above chance, so this again does not estimate forgetting.

This seed demonstrates material candidate-length sensitivity. Under the frozen
summed-log-probability rule, the turn policy selected `ready` for half of far
items, while the secondary normalized rule selected the longer wrong reply.
For clarification, the summed rule selected the long correct reply on 51 of
128 items, whereas normalization selected it on all 128. The primary metric is
not changed after observing this result; both views must be reported.

Across the first two corrected IID seeds, truth and property transfer are the
most consistent positive signals. Entity reference remains near chance.
Counting and the two length-sensitive conventions vary substantially. A third
IID seed is still required before the frozen condition summary.

## Corrected IID CLM seed 4002 and condition summary

Seed 4002 completed 5,002,094 visible tokens across 271 logical rollouts. Its
retention baseline was recorded at 4,502,446 tokens.

| Skill | Independent | Far | Length-normalized far |
|---|---:|---:|---:|
| Turn-taking | 0.0% | 0.0% | 0.0% |
| Truth judgment | 50.8% | 53.1% | 53.1% |
| Entity reference | 13.3% | 21.9% | 14.1% |
| Property binding | 53.1% | 47.7% | 47.7% |
| Counting | 45.3% | 14.1% | 14.1% |
| Clarification | 50.0% | 0.0% | 100.0% |

Primary retention scores changed from 0% to 0% for turn-taking and from 19.5%
to 22.7% for entity reference. Length-normalized entity scores changed from
25.0% to 15.6%. As in the other IID seeds, the held-out skills did not show
robust above-chance acquisition at the retention boundary.

All three corrected IID seeds are now complete:

| Skill | Primary far mean ± sample SD | Length-normalized far mean ± sample SD |
|---|---:|---:|
| Turn-taking | 16.7% ± 28.9% | 0.0% ± 0.0% |
| Truth judgment | 65.9% ± 12.5% | 65.9% ± 12.5% |
| Entity reference | 20.6% ± 2.3% | 17.2% ± 2.7% |
| Property binding | 55.7% ± 7.5% | 55.7% ± 7.5% |
| Counting | 25.3% ± 13.8% | 25.3% ± 13.8% |
| Clarification | 46.6% ± 50.3% | 100.0% ± 0.0% |

The three-seed IID result supports strong transfer for property binding and
moderate transfer for truth judgment. Entity reference remains close to its
16.7% chance level. Counting falls from 54.7% mean independent accuracy to
25.3% far accuracy, consistent with substantial wording/family dependence.
Turn-taking fails under length normalization. Clarification is wholly
scoring-rule-sensitive: the normalized rule always selects its long canonical
reply, while the frozen primary rule is highly variable.

Pre-holdout primary accuracy averaged 0% for turn-taking and 15.4% for entity
reference. Because neither retained skill was acquired robustly above chance,
the IID condition does not provide a meaningful forgetting estimate. This is a
completed within-condition development summary, not evidence for or against a
caregiver-RL advantage.

## Ordered CLM progress

Ordered CLM seed 4000 completed 5,002,265 visible tokens across 277 logical
rollouts. Its retention baseline was recorded at 4,500,511 tokens.

| Skill | Independent | Far | Length-normalized far | Paired primary far minus IID |
|---|---:|---:|---:|---:|
| Turn-taking | 0.0% | 0.0% | 0.0% | 0.0 pp |
| Truth judgment | 71.9% | 78.1% | 78.1% | 0.0 pp |
| Entity reference | 13.3% | 21.9% | 18.0% | +3.9 pp |
| Property binding | 33.6% | 45.3% | 45.3% | −17.2 pp |
| Counting | 24.2% | 22.7% | 22.7% | +1.6 pp |
| Clarification | 0.0% | 50.0% | 100.0% | −50.0 pp |

The clarification difference disappears under length normalization because
both paired models then select the long canonical reply on every far item.
Turn and entity were below chance at the ordered pre-holdout baseline, so this
seed does not estimate forgetting. One paired seed does not support a
condition-level ordering; ordered seeds 4001 and 4002 remain.
