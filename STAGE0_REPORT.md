# Stage-0 engineering report

Date: 2026-07-26  
Status: optimizer gate passed; scientific matrix not started

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
