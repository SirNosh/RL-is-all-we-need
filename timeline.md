# Experiment timeline

## 2026-07-26

- Began by treating the first uploaded note as the research rationale and the second as the concrete experiment specification.
- Chose to inspect the environment before implementing so the run can be scaled to the available hardware without silently changing the scientific comparison.
- Found an empty workspace, a CPU-only default Python, and a separate CUDA
  PyTorch installation. Used the existing CUDA installation after GPU resources
  were freed; installed only the missing SentencePiece binding.
- Implemented the minimum executable six-condition comparison in one file,
  covering all 36 competencies and disjoint train/far surface forms.
- Invalidated the first 50k-token smoke result after discovering the model was
  missing the protocol's RoPE position encoding. Added RoPE and a forward-shape
  regression test before collecting the reported run.
- Completed 18 corrected pilot runs: six conditions, three paired seeds, and
  500k visible tokens per condition. Adaptive RL averaged 23.76% versus 31.05%
  for IID CLM, and Stage-0 adaptive-RL scores remained near chance.
- Decided not to launch the 60-run, 3-billion-token matrix because the
  implementation failed the engineering-gate directionally and still lacks the
  complete caregiver/PPO mechanics. Scaling it would consume compute without
  producing a valid test of the locked hypothesis.
- Accepted the external review's narrower verdict: the legacy run was a sparse
  four-action contextual-bandit test, not an artificial-childhood experiment.
- Replaced the active study with six Stage-0 competencies and eight independent
  generator families, preserving the old source and raw result for audit.
- Isolated the new tokenizer to training families; printable reserve strings
  provide enough BPE merges without importing evaluation names or prompts.
- Implemented short textual response candidates, contingent correction,
  analogous retries, GAE, frozen rollout probabilities, two clipped PPO epochs,
  diagnostic mastery gates, and 20 percent review.
- The mandatory 1,000-situation entity-reference overfit gate passed at 98.5
  percent after seven 512-dialogue rollouts. This licenses bounded development
  testing, not a confirmatory scientific claim.
- Added atomic rollout-boundary checkpoints and traces. A CPU continuous versus
  split run produced bit-identical parameters and curriculum state.
- Resource calibration reduced peak allocated VRAM from roughly 11.8 GB to
  5.7 GB by lowering candidate-scoring chunk size.
- Calibration exposed excessive PPO drift. Froze a development setting of
  3e-5 policy learning rate with a 0.03 target-KL safeguard; the overfit gate
  re-passed at 95.5 percent after 68 rollouts.
- Engineering seed 3200 completed 516,170 tokens with successful CUDA resume
  but failed coverage: only turn, truth, and entity were sampled. Added a
  logged 5 percent frontier-probe allocation without relaxing mastery status.
- Engineering seed 3201 passed the mechanism-operation gate at 506,795 tokens:
  all six skills were exposed, retries and five diagnostics executed, review
  was 19.4 percent of opportunities, and no numerical/CUDA failure occurred.
- Froze development seeds 4000, 4001, and 4002, and added a final 500k-token
  retention holdout for turn-taking and entity reference.
- IID CLM development seed 4000 completed 5,000,756 tokens. Truth and
  clarification transferred well, while final post-holdout turn/entity scores
  were weak. Because no pre-holdout baseline was recorded, this does not
  measure forgetting. Seed 4001 stopped at 36,830 tokens and has no endpoint.
- Paused the remaining matrix after review identified two protocol defects:
  matrix-level configuration hashing made single-run resume invalid, and the
  retention result lacked a pre-holdout baseline.
- Replaced Git-SHA resume validation with hashes of the tokenizer, executable
  experiment/model files, pinned requirements, and an explicit per-run
  scientific configuration. Session controls and outer matrix lists are
  deliberately excluded.
- Added pre/post retention evaluation on distinct seed bands, a dedicated
  retention-start checkpoint, actual holdout-start token logging, graceful
  session exits, the frozen 3e-5 PPO default, and candidate length/prior
  diagnostics without changing the primary summed-log-probability rule.
- Preserved v1 seed 4000 as `development_v1_pre_retention_baseline` and the
  36,830-token seed 4001 checkpoint as `interrupted_unresumable_under_v1`.
- Published the corrected protocol as commit `87c5773` on draft PR #1; all 13
  CPU tests passed.
- An initial corrected-run launcher accidentally selected CPU-only PyTorch and
  was stopped after 55,784 tokens. A second launcher lacked SentencePiece and
  exited before training. Both were invalidated; the scientific run restarted
  from zero under the verified Miniconda CUDA environment.
- Corrected IID CLM seed 4000 completed at 5,000,756 tokens after one clean
  session-limit resume. The retention baseline occurred at 4,501,006 tokens.
  Turn was 0% before and after withholding; entity was 9.375% before and
  16.406% after. Neither pre-score exceeded chance, so this measures weak
  acquisition rather than forgetting.
- The final corrected acquisition endpoint matches the v1 headline pattern:
  truth 78.1%, property 62.5%, and clarification 100% far accuracy; turn 0%,
  entity 18.0%, and counting 21.1%. Length-normalized scoring changed only
  entity far accuracy, from 18.0% to 18.8%.
- Corrected IID CLM seed 4001 completed at 5,002,499 tokens after one clean
  resume. Primary far accuracy was turn 50.0%, truth 66.4%, entity 21.9%,
  property 57.0%, count 40.6%, and clarification 39.8%.
- Seed 4001 exposed material scoring-rule sensitivity: length normalization
  changed turn from 50% to 0% and clarification from 39.8% to 100%. Preserved
  summed log probability as the frozen primary rule and reported both rather
  than changing the metric after observing the endpoint.
- Turn/entity were again not meaningfully above chance before withholding, so
  the second corrected run also provides no estimate of forgetting.
- Corrected IID CLM seed 4002 completed at 5,002,094 tokens after recording its
  retention baseline at 4,502,446. Primary far accuracy was turn 0%, truth
  53.1%, entity 21.9%, property 47.7%, count 14.1%, and clarification 0%.
  Clarification was again 100% under length normalization.
- Completed the corrected three-seed IID condition. Mean primary far accuracy
  was turn 16.7%, truth 65.9%, entity 20.6%, property 55.7%, count 25.3%, and
  clarification 46.6%. The normalized means were 0%, 65.9%, 17.2%, 55.7%,
  25.3%, and 100%, respectively.
- Concluded only that IID CLM shows heterogeneous acquisition: truth/property
  transfer, near-chance entity reference, poor far counting relative to
  independent items, and severe candidate-length sensitivity for the two
  conventional replies. Cross-condition evidence still does not exist.
- Ordered CLM seed 4000 completed at 5,002,265 tokens. Relative to paired IID
  seed 4000, primary far differences were 0 pp turn, 0 pp truth, +3.9 pp
  entity, −17.2 pp property, +1.6 pp count, and −50 pp clarification.
- The paired clarification gap vanishes under length normalization (both
  100%), reinforcing that candidate scoring must accompany all condition
  comparisons. Deferred any condition verdict until ordered seeds 4001–4002.
- Ordered CLM seed 4001 completed at 5,002,787 tokens. Paired primary far
  differences from IID were 0 pp turn, +11.7 truth, 0 entity, −2.3 property,
  −26.6 count, and +60.2 clarification; the clarification gap was again zero
  under length normalization.
- Primary turn accuracy was 50% before and after withholding, but normalized
  accuracy was 0% at both points. Recorded the nominal retention ratio of 1.0
  while treating it as scoring-rule-sensitive rather than robust retention.
- Ordered CLM seed 4002 completed at 5,000,103 tokens, completing the
  three-seed ordered condition.
- Paired ordered-minus-IID primary far differences averaged 0 pp turn, +15.1
  truth, −0.8 entity, +2.9 property, −6.8 count, and +36.7 clarification.
  Under length normalization the clarification difference was 0 for all seeds.
- Treated improved truth judgment as the only reasonably consistent first
  cross-condition signal. Property/count contrasts varied by seed, entity
  stayed near chance, and retention remained uninterpretable because the held
  skills were not robustly acquired before withholding.
- Fixed caregiver RL seed 4000 completed at 5,019,278 tokens across 298 PPO
  rollouts and three bounded sessions. Far accuracy was turn 100%, truth 78.1%,
  entity 13.3%, property 49.2%, count 23.4%, and clarification 0%.
- Turn remained 100% before and after a 489,573-token holdout under both
  scoring rules, the first robust retention observation for a prespecified
  skill. Entity was near chance and did not support a forgetting estimate.
- Corrected the RL resource forecast after entity-stage PPO peaked near 10 GB
  allocated and essentially filled the 12 GB card. Reserved subsequent
  caregiver-RL sessions exclusively up to 11 GB physical VRAM rather than
  changing frozen minibatching mid-comparison.
- The fixed run completed without numerical/CUDA failure, but target-KL early
  stopping fired on 187/298 rollouts and the maximum rollout-mean KL was 0.284;
  retained this as a monitoring item rather than recalibrating after endpoints
  had begun.
- Fixed caregiver RL seed 4001 completed at 5,040,652 tokens across 308 PPO
  rollouts and four resumed sessions. Far accuracy was turn 100%, truth 78.1%,
  entity 14.8%, property 49.2%, count 25.8%, and clarification 0% primary/100%
  normalized.
- Turn was again 100% before and after withholding under both scoring rules.
  This reproduces the fixed-caregiver turn acquisition/retention signal in a
  second seed, while entity and the other far skills do not show broad RL
  superiority.
- PPO remained finite but target-KL early stopping fired on 209/308 rollouts;
  mean rollout KL was 0.0237 and the maximum was 0.314.
- Fixed caregiver RL seed 4002 completed at 5,038,048 tokens across 271 PPO
  rollouts, completing the three-seed fixed condition. Far accuracy was turn
  100%, truth 61.7%, entity 15.6%, property 39.1%, count 27.3%, clarification
  0%.
- The complete fixed-caregiver condition acquired and retained turn-taking at
  100% in 3/3 seeds under both scoring rules. Paired primary differences from
  IID averaged +83.3 pp turn, +6.8 truth, −6.0 entity, −9.9 property, +0.3
  count, and −46.6 clarification.
- Interpreted this as a narrow positive interaction result, not broad support
  for artificial childhood: caregiver contingency reliably teaches the
  interaction-specific turn convention but does not improve the wider skill
  suite. Entity stays near/below chance and later-syllabus skills show tradeoffs.

## 2026-07-28 12:51 EDT

- Began the adaptive-caregiver condition with seed 4000 using the frozen
  scientific configuration. Two bounded sessions completed cleanly at
  1,549,121 and 3,019,960 visible tokens; the third session is in progress.
- Preserved session boundaries so the shared RTX 4070 SUPER can be released
  between resumptions. Each GPU acquisition and release is recorded in the
  append-only shared resource ledger after checking both that ledger and live
  GPU use.
- The first session produced 90 PPO rollouts, 15 diagnostics, and 2,280
  frontier probes. Turn-taking was mastered; truth and entity became eligible;
  no OOM, CUDA, NaN, or trace failure occurred. No comparative interpretation
  will be made until the seed reaches its endpoint, and no condition-level
  claim until all three paired seeds complete.
- Adaptive caregiver seed 4000 completed at 5,007,180 tokens across 341
  rollouts and four bounded sessions. Far accuracy was 100% turn, 66.4% truth,
  19.5% entity, 43.8% property, 15.6% count, and 0% clarification.
- Turn was acquired early and retained at 100% under both scoring rules. Truth
  was marked mastered only at 4.31M tokens. Entity was not robustly above
  chance at the retention baseline, so it does not support a forgetting claim.
- Recorded the result as one adaptive seed only. Its paired primary differences
  from IID were +100, âˆ’11.7, +1.5, âˆ’18.7, âˆ’5.5, and âˆ’100 percentage points
  across the six skills; this does not yet identify an adaptive effect.
- Audited 341 finite rollouts: 301 target-KL early stops, mean KL 0.0180,
  maximum KL 0.263, 25,426 retries, 8,638 frontier probes, and no
  NaN/OOM/CUDA failure.
