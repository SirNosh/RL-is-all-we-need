from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from phase1_data import CONCEPTS, RESPONSE_VOCAB, CaregiverLanguage, Lesson, Tokenizer, make_lesson
from phase1_model import Student, candidate_scores


@dataclass
class Transition:
    context: str
    action: int
    reward: float
    old_logp: float
    old_value: float
    dialogue: int
    advantage: float = 0.0
    return_: float = 0.0


def centered_reward(action: int, target: int) -> float:
    return 1.0 if action == target else -1.0 / (len(RESPONSE_VOCAB) - 1)


def feedback(condition: str, lesson: Lesson, rng: random.Random, caregiver: CaregiverLanguage):
    if condition == "contingent_caregiver_rl":
        return lesson.correction, lesson.analogous_prompt, lesson.analogous_target
    if condition == "yoked_caregiver_rl":
        other = rng.choice([c for c in CONCEPTS if c != lesson.concept])
        other_lesson = make_lesson(other, rng.randrange(1_000_000), rng.randrange(4), caregiver)
        text = f"Caregiver: look at another thing.\n{other_lesson.visible_text}\nCaregiver: this is {other}."
        return text, lesson.analogous_prompt, lesson.analogous_target
    return "", lesson.analogous_prompt, lesson.analogous_target


def collect_rollout(model: Student, tok: Tokenizer, condition: str, rng: random.Random, lessons: list[Lesson], caregiver: CaregiverLanguage, device):
    contexts = [x.prompt if condition == "trial_only_rl" else x.demonstration + "\n" + x.prompt for x in lessons]
    scores, values = candidate_scores(model, tok, contexts, RESPONSE_VOCAB, device)
    dist = torch.distributions.Categorical(logits=scores)
    actions = dist.sample(); logps = dist.log_prob(actions)
    transitions, retry_rows = [], []
    visible, retries, attempt1, eventual = 0, 0, 0, 0
    choices, rewards = Counter(), defaultdict(list)
    for i, lesson in enumerate(lessons):
        target, action = RESPONSE_VOCAB.index(lesson.target), int(actions[i])
        reward = centered_reward(action, target)
        transitions.append(Transition(contexts[i], action, reward, float(logps[i].detach().item()), float(values[i].detach().item()), i))
        visible += len(tok.encode(contexts[i])) + len(tok.encode(" " + RESPONSE_VOCAB[action]))
        choices[(lesson.concept, RESPONSE_VOCAB[action])] += 1; rewards[lesson.concept].append(reward)
        if action == target:
            attempt1 += 1; eventual += 1; continue
        if condition == "trial_only_rl":
            continue
        teach, retry_prompt, retry_target = feedback(condition, lesson, rng, caregiver)
        retry_context = contexts[i] + " " + RESPONSE_VOCAB[action] + "\n" + teach + "\n" + retry_prompt
        retry_rows.append((i, retry_context, retry_target))
        visible += len(tok.encode("\n" + teach + "\n" + retry_prompt)); retries += 1
    if retry_rows:
        rscores, rvalues = candidate_scores(model, tok, [x[1] for x in retry_rows], RESPONSE_VOCAB, device)
        rdist = torch.distributions.Categorical(logits=rscores)
        ractions = rdist.sample(); rlogps = rdist.log_prob(ractions)
        for j, (dialogue, context, target_text) in enumerate(retry_rows):
            target, action = RESPONSE_VOCAB.index(target_text), int(ractions[j])
            reward = .6 if action == target else -1.0 / (len(RESPONSE_VOCAB) - 1)
            transitions.append(Transition(context, action, reward, float(rlogps[j].detach().item()), float(rvalues[j].detach().item()), dialogue))
            visible += len(tok.encode(" " + RESPONSE_VOCAB[action])); eventual += int(action == target)
    metrics = {
        "dialogues": len(lessons), "transitions": len(transitions), "retries": retries,
        "attempt1_accuracy": attempt1 / len(lessons), "eventual_success": eventual / len(lessons),
        "candidate_counts": {f"{a}::{b}": n for (a, b), n in choices.items()},
        "mean_reward_by_concept": {k: float(np.mean(v)) for k, v in rewards.items()},
    }
    return transitions, visible, metrics


def assign_gae(transitions, gamma=.99, lam=.95):
    groups = defaultdict(list)
    for transition in transitions:
        groups[transition.dialogue].append(transition)
    for sequence in groups.values():
        gae, next_value = 0.0, 0.0
        for transition in reversed(sequence):
            delta = transition.reward + gamma * next_value - transition.old_value
            gae = delta + gamma * lam * gae
            transition.advantage, transition.return_ = gae, gae + transition.old_value
            next_value = transition.old_value


def ppo_update(model, tok, optimizer, transitions, device, rng, epochs=2, minibatch=64, clip=.2, value_coef=.5, target_kl=.03):
    assign_gae(transitions)
    advantages = torch.tensor([x.advantage for x in transitions], device=device)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    indices, kls, clips, losses, early = list(range(len(transitions))), [], [], [], False
    for _ in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), minibatch):
            selected = indices[start:start+minibatch]; batch = [transitions[i] for i in selected]
            scores, values = candidate_scores(model, tok, [x.context for x in batch], RESPONSE_VOCAB, device)
            dist = torch.distributions.Categorical(logits=scores)
            actions = torch.tensor([x.action for x in batch], device=device)
            new = dist.log_prob(actions); old = torch.tensor([x.old_logp for x in batch], device=device)
            returns = torch.tensor([x.return_ for x in batch], device=device); adv = advantages[torch.tensor(selected, device=device)]
            ratio = (new - old).exp(); objective = torch.minimum(ratio * adv, ratio.clamp(1-clip, 1+clip) * adv)
            loss = -objective.mean() + value_coef * .5 * (values - returns).pow(2).mean() - .01 * dist.entropy().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            kl = .5 * (new.detach() - old).pow(2).mean().item()
            kls.append(kl); clips.append(((ratio.detach() < 1-clip) | (ratio.detach() > 1+clip)).float().mean().item()); losses.append(loss.item())
            if kl > target_kl:
                early = True; break
        if early: break
    return {"loss": float(np.mean(losses)), "approx_kl": float(np.mean(kls)), "max_minibatch_kl": float(max(kls)), "clip_fraction": float(np.mean(clips)), "early_stopped_for_kl": early}
