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
