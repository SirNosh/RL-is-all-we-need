# Textual Developmental Pretraining

This repository is a compact executable engineering pilot for the uploaded
Experiment 1 protocol. It preserves the central controlled comparison:

- the same random decoder architecture and tokenizer in every condition;
- 36 competencies spanning language, mathematics, reasoning, science,
  geography, and coding;
- deterministic training and disjoint far-transfer surface forms;
- IID, ordered, and adaptive CLM;
- fixed and adaptive interactive RL;
- adaptive hybrid training;
- matched learner-visible token budgets and paired initialization seeds.

It intentionally does not claim that a short pilot is the locked confirmatory
experiment. The uploaded protocol requires 60 runs of 50 million tokens plus
retention and learning-to-learn branches. That is a substantial compute study,
not a one-session run.

## Reproduce the executed pilot

```powershell
python -m pip install -r requirements.txt
python experiment.py --budget 500000 --seeds 1000 1001 1002
```

Results are written incrementally to `results/`, so an interrupted run retains
completed conditions.

The reported run used Python 3.12, PyTorch 2.6.0+cu124, NumPy 2.3.1,
SentencePiece 0.2.0, and an RTX 4070 Super. Reviewers with an NVIDIA GPU should
install the PyTorch wheel appropriate to their CUDA runtime before installing
the remaining requirements.

## Run the locked core training matrix

The following launches the prescribed six conditions and ten paired seeds:

```powershell
python experiment.py --budget 50000000 --seeds 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009
```

The compact implementation is an engineering instrument, not yet the complete
preregistered study: it uses deterministic multiple-choice responses to make
RL bootstrap measurable, and it does not yet implement delayed retention,
capstones, learning-to-learn branches, 100,000-sample generator validation, or
the prescribed permutation analysis. Those omissions must be closed before
results can support the scientific claim.
