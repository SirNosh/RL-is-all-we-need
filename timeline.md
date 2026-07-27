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
