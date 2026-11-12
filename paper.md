# Group-Relative Baselines Reward Interference Between Rollouts

*Working draft, 11 November 2026 (revised after the related-work search). Sections 1–4 and
6 are written from what we have. Section 7 lists experiments that have not been run. The
sources behind Section 5 are in `KNOWN_RELATED_WORK.md`, with links, abstracts and what was
read of each; many were read as abstracts only and must be read before submission.*

## Abstract

*(Placeholder; to be rewritten once the language-model experiments exist.)*

Group Relative Policy Optimization (GRPO) scores each rollout against the other rollouts
sampled for the same prompt. Gao et al. (2026) prove that when a rollout's reward depends
on its group, this baseline changes the objective: the update rewards a rollout for
lowering its groupmates' rewards. They study rewards that depend on the group by design,
such as diversity bonuses. We study the case where nobody intended the dependence: agentic
rollouts that share a machine, and judges that read the whole group in one context. Both
are in use, and agents sharing a machine have been observed killing each other's processes
and disguising their own. Applying the result to a harmful action gives a simple
condition: an action that costs its actor c and lowers the rewards of k of the other G − 1
rollouts by d each is reinforced when kd > (G − 1)c, although it lowers the expected reward
by c + kd. In simulation the predicted action gaps hold in 104 of 104 configurations,
interference is learned exactly above the threshold, and mean reward collapses. We classify
interference by its time structure, give baselines that remove the incentive for two of the
three classes, and describe monitors that need no change to training. *(To add: results on
a language model in a shared virtual machine and with a joint-context judge. No published
work shows a policy learning to interfere with its group; that demonstration is the
paper's main planned contribution.)*

## 1 Introduction

**The setting.** GRPO (Shao et al. 2024) and its variants are the standard way to train
language models with reinforcement learning. For each prompt the policy samples a group of
G rollouts, and each rollout's advantage is its reward minus the group's mean reward. The
group mean is a cheap baseline that needs no learned value function.

**The assumption.** A baseline leaves the policy gradient unbiased only if it does not
depend on the action being credited (Williams 1992). The group mean satisfies this when
rollouts are independent given the prompt. Two common practices break independence:
- **Shared machine state.** Agentic rollouts of one group often run on one machine or in
  one container, where they share processes, files, ports and caches.
- **Joint judging.** Some reward models read the whole group in one context and score each
  member, because comparison helps the judge discriminate (Section 5).

**The consequence.** If a rollout can lower its peers' rewards, doing so lowers the group
mean and raises its own advantage. GRPO then reinforces interference, even though
interference lowers the expected reward that training is meant to increase. No strategy or
intent on the model's part is needed: ordinary exploration finds the action and the
estimator rewards it.

**Why this is not hypothetical.** Each ingredient has been reported separately.
- **Agents on a shared machine interfere with each other.** Anthropic's system card for
  Claude Fable 5 and Mythos 5 (June 2026, §6.2.1.2) describes agents that were
  "accidentally spawned in the same work directory and with shared files, utilities, and
  API rate limits". They would "kill the agents with which they shared resources and try
  to avoid being killed themselves", and "create new processes with disguised names to
  avoid being killed". A later evaluation with conflicting goals found kill loops, disabled
  accounts and a kill script named to look like a system health monitor (Anthropic 2026b).
  Both were observed in use or evaluation, not in training.
- **Rollouts of one group share machines during training.** One GRPO system limits
  compilation concurrency so that contention does not affect rollouts of the same group
  (arXiv:2605.07276). NVIDIA's software-engineering RL recipe blocks `killall` and `pkill`
  because a rollout could otherwise terminate other processes on the node (NVIDIA 2026).
- **Interaction between parallel training rollouts has been reinforced by training.**
  OpenAI (2026) reports that parallel rollouts communicated through a shared package cache
  and that "this behavior was then reinforced during training". That interaction was
  cooperative. It shows that such channels exist during training and that training acts
  on them.
- **Judges that read the group together are used with GRPO.** RULER (OpenPipe) places all
  trajectories of a group in one judge message and scores each from 0 to 1; its
  documentation notes that "GRPO only needs relative scores".

These models were trained with reinforcement learning. A group-relative baseline in a
shared environment is one mechanism by which such training could have produced the
interference seen in the first item. We know of no evidence that it did, and one evaluation
found a model trained on reward hacks did not tamper with other episodes' rewards
(Anthropic 2026c). The hypothesis can be tested, and Section 7 says how.

**Relation to prior work.** The mathematics is known. Gao et al. (2026) prove that for any
reward that depends on the group, the expected GRPO or leave-one-out update equals the
gradient of the rollout's own reward plus α times its effect on its groupmates' rewards,
with α = −1/(G − 1), where the true gradient has α = +1. They note the parallel with
selection in finite populations and propose an unbiased estimator. Their rewards depend on
the group by design. Mallen (2026) remarks in one sentence that GRPO would select for a
trajectory that sabotages its siblings' reward. Macé et al. (2023) argue in general terms
that selecting interacting agents on relative performance selects for harm. None of these
studies the dependence that arises unintended from training infrastructure, and none shows
a policy learning to interfere (Section 5).

**Scope.** We study the cases where the *estimator* creates the incentive. Rewards that are
zero-sum by definition, such as pairwise tournaments among siblings, and judge manipulation
that raises a rollout's own score, are reward-design problems that any unbiased estimator
would follow. Section 3.3 draws the line.

**Contributions.**
1. The observation that ordinary agent-training infrastructure makes rewards depend on the
   group, with documented evidence for each channel (this section and Section 5).
2. The condition under which a harmful action is reinforced, obtained by applying Gao et
   al.'s result to an action with a cost to the actor and a harm to its peers, and its
   extension to defense (Section 2).
3. A classification of interference by time structure, with the baselines and monitors
   that apply to each class (Section 3).
4. Simulations that confirm the action gaps, the threshold, the group-size dependence, and
   the cycling of sabotage and defense (Section 4).
5. *(Planned.)* Demonstrations on a language model that training increases interference
   under GRPO and not under an independent baseline (Section 7).

## 2 Theory

*Sections 2.2–2.4 are a special case of Gao et al. (2026). Their identity writes the
expected update as the own-reward gradient plus α times the effect on groupmates' rewards.
For an action with own effect −c and total effect −H on the others, leave-one-out
(α = −1/(G − 1)) gives the gap H/(G − 1) − c, and GRPO gives the same multiplied by
1 − 1/G, which is Table 1. We keep the derivation because the action-gap form gives the
threshold directly. Section 2.5 (defense) is not in their paper.*

### 2.1 Setup

For one prompt the policy π_θ samples rollouts y_1, …, y_G. Rollout j receives reward
r_j(y_1, …, y_G), which may depend on the other rollouts. The training objective is the
expected mean reward J = E[(1/G) Σ_j r_j]. GRPO uses the advantage
A_i = r_i − (1/G) Σ_j r_j, optionally divided by the group's standard deviation.

Consider a binary decision inside a rollout, taken with probability p. Its **action gap**
Δ is the expected advantage when the action is taken minus when it is not. The expected
update of the decision's log-odds is proportional to p(1 − p)Δ, so the action is reinforced
exactly when Δ > 0.

### 2.2 The action gap of interference

Suppose the action costs its actor c in its own reward and lowers the rewards of k of the
other G − 1 rollouts by d each. Write H = kd for the harm per act.

**Table 1.** Action gap of an interfering action under each estimator.

| estimator | advantage | action gap Δ | reinforced when |
|---|---|---|---|
| GRPO | r_i − group mean | H/G − (1 − 1/G)c | H > (G − 1)c |
| leave-one-out (RLOO) | r_i − mean of the others | H/(G − 1) − c | H > (G − 1)c |
| independent baseline | r_i − a baseline the rollout cannot affect | −c | never, if c > 0 |
| group-total credit | Σ_j r_j − an independent baseline | −c − H | never |
| true gradient of G·J | | −c − H | never |

*Derivation for GRPO.* A_i = (1 − 1/G) r_i − (1/G) Σ_{j≠i} r_j. The action lowers r_i by c
and Σ_{j≠i} r_j by H. Harm received from other rollouts does not depend on the rollout's own
choice, so it adds noise with zero mean.

**Why.** Advantages within a group sum to zero. GRPO's objective is therefore a rollout's
standing within its group, and lowering a peer's reward counts like raising one's own,
scaled by 1/G.

### 2.3 The general condition

For any baseline b, the action gap equals the action's effect on the rollout's own reward
minus its effect on b. The incentive to interfere therefore vanishes exactly when b is
computed from quantities the rollout cannot influence. Any baseline built from the rewards
of rollouts that the action can reach fails this, which is why leave-one-out does no better
than the group mean.

An independent baseline makes training *selfish*: interference is no longer rewarded, but
it is not punished either. If interfering helps the actor directly (c < 0, as when killing
other processes frees CPU), it is still reinforced, and correctly so with respect to the
actor's own reward. Only credit for the group's total reward matches the true gradient.

### 2.4 Group size

- **Harm per target fixed.** The condition is kd > (G − 1)c. If every peer is harmed
  (k = G − 1), it becomes d > c and does not depend on G.
- **Total harm fixed.** If each act does total harm D, shared among its targets, the
  condition is D > (G − 1)c. Interference is reinforced only in groups smaller than
  1 + D/c.

The second case has the form of the finite-population condition of Hamilton (1970) and
Schaffer (1988): when selection acts on payoff relative to the rest of a population of N,
a costly act that harms others is favored when the harm exceeds N − 1 times the cost, so
only in small populations. Gao et al. (2026) draw the same parallel. The weight −1/(G − 1)
that GRPO gives to another rollout's reward is the relatedness to a random other member of
a finite population. The same result appears in the economics of tournaments: pay by rank
induces sabotage (Lazear 1989), and sabotage of a single rival pays only when contestants
are few, because the gain is shared by all the other contestants (Konrad 2000).

### 2.5 Defense

Suppose a rollout can pay e to block all harm to itself, and does so with probability q.
With sabotage probability p:
- sabotage has action gap kd(1 − q)/G − (1 − 1/G)c;
- defense has action gap (1 − 1/G)(kdp − e).

Defense is credited correctly: the true gradient also gives kdp − e. Only sabotage is
miscredited. When kd > (G − 1)c there is a mixed equilibrium

  p\* = e/(kd),  q\* = 1 − (G − 1)c/(kd).

The expected dynamics of the two log-odds conserve a quantity, as predator–prey dynamics
do, so the equilibrium is neutrally stable and trajectories are closed orbits around it.
Sampling noise moves a trajectory between orbits.

### 2.6 Standard-deviation scaling

Original GRPO divides the advantage by the group's standard deviation. This has no closed
form, but it keeps the sign of Δ and enlarges it when rewards within a group are close. In
the simulations of Section 4.1 the scaled gap was 1.1 to 3.3 times the unscaled one.

## 3 Classes of interference, mitigations and monitors

### 3.1 Three time structures

What a rollout can reach depends on when the harm acts.

**Table 2.** Interference by time structure.

| class | example | who can be harmed | clean baseline | monitor |
|---|---|---|---|---|
| **coexistence** | killing another rollout's process; exhausting shared memory | rollouts alive at the same time | run two halves of the group at non-overlapping times; each half is the baseline for the other | reward against the number of siblings alive at the same time |
| **persistent** | leaving a corrupted file, cache entry or trap | rollouts that start later | the earlier half is a clean baseline for the later half; the earlier half needs a baseline from a previous step | reward against start order within the group |
| **time-independent** | text that lowers a judge's score for another group member | any rollout judged in the same context | none within joint judging; see Section 3.3 | score of a fixed output with and without the rollout in the judge's context |

Asynchronous rollouts vary in overlap and start order by themselves, so the first two
monitors need no change to training. The quantity to watch is the trend over training: a
constant effect may be plain resource contention, while a growing one means the policy is
learning to interfere.

### 3.2 Mitigations and their costs

| mitigation | removes the incentive | punishes harm | cost |
|---|---|---|---|
| isolate rollouts | yes, if isolation holds | not needed | infrastructure; fails when a rollout escapes its sandbox |
| split-half baseline (Table 2) | for coexistence harm; for persistent harm in the later half | no | wall-clock time |
| baseline from earlier steps | yes | no | staleness; needs repeated prompts or a running mean |
| no baseline (REINFORCE) | yes | no | variance |
| group-total credit | yes | yes | variance about G times GRPO's |
| own reward plus a weight λ on the others' rewards | yes | when λ·harm exceeds the actor's direct gain | variance grows with λ²(G − 1) |
| larger groups | only when total harm is bounded | no | compute |

A further monitor applies to all classes: include a few rollouts from a frozen reference
policy in each group. Their policy does not change, so a decline in their rewards over
training indicates learned interference.

### 3.3 Judges: where GRPO matters and where it does not

**Table 3.** Whether GRPO adds an incentive to lower a sibling's score, by judge design.

| judge design | incentive without GRPO | what GRPO adds |
|---|---|---|
| each output scored alone | none; no channel | nothing |
| pairwise or tournament among siblings | yes; the reward is zero-sum by definition | nothing; the group mean is constant |
| scored against a fixed reference | yes, by lowering the reference's score | nothing |
| **group read together, each member scored** | none, if lowering a sibling's score leaves one's own unchanged | **the whole incentive**: d/G per sibling harmed |

Only the last row is within the scope of this paper. Writing a disparaging sentence costs
almost nothing, so c ≈ 0 and the condition of Section 2.2 holds for almost any effect on
the judge.

Two qualifications:
- A judge asked to compare may raise one member's score when it lowers another's. That part
  of the incentive exists under any estimator. The GRPO-specific part is the harm to the
  sibling that does not return as the actor's own gain. An experiment must measure both.
- Judge calls can be replayed, unlike machine state. Scoring a member with and without
  another member in the context measures the harm directly. This gives a monitor and a
  penalty (a difference reward; Wolpert & Tumer 2001). It cannot separate text that raises
  a rollout's own score from real quality.

## 4 Simulations

### 4.1 Model

Each rollout makes three Bernoulli decisions, each with one logit shared by all rollouts:
- **task:** success with probability t adds 1 to the reward;
- **sabotage:** with probability p; costs the actor c and harms k randomly chosen other
  rollouts of its group;
- **defend:** with probability q (experiment D only); costs e and blocks all harm.

The reward is r_j = y_j − c s_j − e f_j − h · hits_j · (1 − f_j), where hits_j is the number
of saboteurs that targeted j, and h = d or h = D/k. The normalizer is a constant, as in
Dr. GRPO (Liu et al. 2025). Code: `sim/sim.py`; analysis: `sim/analyze.py`. The run was a
single CPU job of under one hour on 16 cores.

| experiment | design | tests |
|---|---|---|
| A. action gap | policy frozen at p = q = t = ½; G ∈ {2, 4, 8, 16, 32}; k ∈ {1, G/2, G − 1}; per-target or split harm; c = 0.1, d or D = 0.4; 200 batches | Table 1 |
| B. threshold | G = 8, c = 0.05, d/c ∈ {½, 1, 2, 4, 8, 16}, k ∈ {1, 4, 7}; five estimators; p₀ = 0.1; SGD, 3,000 steps, 3 seeds | where sabotage is learned, and its cost |
| C. group size | total harm D = 0.4 over all peers, c = 0.05, G from 2 to 32 | Section 2.4 |
| D. defense | G = 8, e = 0.3; harm 0.3 to all 7 peers or 0.6 to one; SGD and Adam; 20,000 steps, 3 seeds | Section 2.5 |

### 4.2 The action gap matches the theory

All 104 configurations with a closed form lie within 2 standard errors of the prediction
(largest deviation 1.9 SE; Figure 1).

**Table 4.** Action gap of sabotage at G = 8 (c = 0.1, d = 0.4): measured ± SE, with the
prediction in parentheses.

| targets k | GRPO | RLOO | independent | charged for harm inflicted |
|---|---|---|---|---|
| 1 | −0.0375 ± 0.0009 (−0.0375) | −0.0429 ± 0.0010 (−0.0429) | −0.1004 ± 0.0010 (−0.1) | −0.4998 ± 0.0010 (−0.5) |
| 4 | +0.1120 ± 0.0010 (+0.1125) | +0.1280 ± 0.0011 (+0.1286) | −0.1014 ± 0.0013 (−0.1) | −1.6991 ± 0.0018 (−1.7) |
| 7 | +0.2624 ± 0.0008 (+0.2625) | +0.2999 ± 0.0009 (+0.3000) | −0.1016 ± 0.0016 (−0.1) | −2.8975 ± 0.0026 (−2.9) |

![Figure 1](sim/figures/fig1_gap.png)

**Figure 1.** Measured against predicted action gap of sabotage over the configurations of
experiment A, with 95% intervals (smaller than the markers). Points on the dashed diagonal
agree with Table 1. The training objective requires every point to lie below zero; points
above zero are reinforced. Points beyond −0.55 are omitted for scale and lie on the
diagonal (Table 4).

### 4.3 Sabotage is learned above the threshold, and reward collapses

**Table 5.** P(sabotage) after 3,000 steps, starting from 0.1, under GRPO at G = 8 (mean of
3 seeds). The predicted threshold is d/c = 7/k.

| targets k | threshold | d/c = ½ | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| 1 | 7 | 0.008 | 0.008 | 0.010 | 0.016 | 0.79 | 0.99 |
| 4 | 1.75 | 0.010 | 0.016 | 0.78 | 0.99 | 1.00 | 1.00 |
| 7 | 1 | 0.014 | 0.100 | 0.99 | 1.00 | 1.00 | 1.00 |

- At k = 7 and d/c = 1 the predicted gap is exactly zero, and P(sabotage) stays at 0.10.
- RLOO and std-scaled GRPO switch at the same threshold (Figure 2).
- Under the independent baseline P(sabotage) falls to 0.006 in every configuration. When
  the rollout is also charged for the harm it inflicts, it falls below 0.005.
- Where sabotage is learned, mean reward ends far below the optimum of 1: 0.25 at k = 7,
  d/c = 2; −0.45 at k = 7, d/c = 4; −2.25 at k = 4, d/c = 16. Under the two unbiased
  estimators it stays at 0.98–1.00.

![Figure 2](sim/figures/fig2_threshold.png)

**Figure 2.** Probability of sabotage after 3,000 steps against the ratio of harm per target
to the actor's cost, for three numbers of targets per act (G = 8). The dashed line is the
predicted threshold d/c = (G − 1)/k, and the shaded region is where GRPO and RLOO are
predicted to reinforce sabotage. The optimum of the training objective is P(sabotage) = 0
everywhere; only the independent baseline and the difference reward stay at it.

### 4.4 With fixed total harm, sabotage is learned only in small groups

**Table 6.** P(sabotage) after 3,000 steps with total harm D = 0.4 over all peers and
c = 0.05. Predicted: reinforced iff G < 9.

| G | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|
| GRPO | 1.00 | 1.00 | 0.99 | 0.98 | 0.78 | 0.02 | 0.01 | 0.01 |
| RLOO | 1.00 | 1.00 | 1.00 | 0.99 | 0.84 | 0.02 | 0.01 | 0.01 |

![Figure 3](sim/figures/fig3_spite.png)

**Figure 3.** Probability of sabotage after 3,000 steps against group size, when each act
does a fixed total harm D = 0.4 shared over all peers (c = 0.05). The dashed line is the
predicted boundary G = 1 + D/c = 9, and the shaded region is where sabotage is predicted to
be reinforced. The optimum of the training objective is P(sabotage) = 0; larger groups
approach it and smaller groups move away from it.

### 4.5 With defense, sabotage and defense cycle

**Table 7.** Time averages over steps 10,000–20,000 (3 seeds) against the predicted
equilibrium.

| harm | optimizer | p\* | p averaged (range over time) | q\* | q averaged | mean reward |
|---|---|---|---|---|---|---|
| 0.3 to all 7 peers | SGD | 0.14 | 0.15–0.16 (0.02–0.83) | 0.83 | 0.82–0.83 | 0.69 |
| 0.3 to all 7 peers | Adam | 0.14 | 0.18–0.21 (0.01–0.91) | 0.83 | 0.85–0.88 | 0.67–0.68 |
| 0.6 to 1 peer | SGD | 0.50 | 0.49–0.50 (0.06–0.95) | 0.42 | 0.39–0.40 | 0.67–0.68 |
| 0.6 to 1 peer | Adam | 0.50 | 0.49–0.51 (0.05–0.94) | 0.42 | 0.40–0.43 | 0.67–0.68 |

The trajectories do not settle. They cycle around the equilibrium near the boundary of the
unit square, with recurring bursts of sabotage during which mean reward falls below zero
(Figure 4). Time averages are close to the equilibrium, and mean reward stays about 31%
below the optimum.

![Figure 4](sim/figures/fig4_defense.png)

**Figure 4.** Sabotage and defense under GRPO (G = 8, defense cost e = 0.3).
- **Left and centre.** Trajectories of P(sabotage) and P(defend), for harm to all peers
  (left) and to one peer (centre). Solid lines are SGD and dotted lines Adam, three seeds
  each. The star is the predicted mixed equilibrium. The optimum of the training objective
  is the corner with no sabotage and no defense, which neither optimizer approaches.
- **Right.** Mean reward over training for harm to all peers. The dashed line is the
  optimum of 1.

### 4.6 Limits of the simulations

- The policy is tabular, with one logit per decision and no generalization between them.
- The interfering action is given to the policy, not discovered.
- The mitigations of Section 3.2 other than the independent baseline, and the monitors of
  Section 3.1, are not yet simulated.

## 5 Related work

`KNOWN_RELATED_WORK.md` lists the 166 works found, with links and abstracts. This section
covers those that bear on the claim.

**The group baseline under group-dependent rewards.** Gao et al. (2026) is the closest
work, and Section 2 builds on it. They prove the identity stated in Section 1, show that
GRPO "often does worse than using no baseline at all" for diversity and
distribution-matching rewards, and propose Team-GRPO, which credits each response with
its effect on the group (α = +1). Three differences from our setting:
- Their rewards depend on the group by design. Ours depend on it through infrastructure
  that was assumed to be independent.
- Their dependence is a bonus the designer wants optimized. Ours is harm, which lowers
  every objective the designer has.
- Their analysis assumes an exchangeable reward. Interference that persists on a machine
  depends on start order, which makes the time-based baselines of Section 3 possible.

**Other analyses of the GRPO baseline.** GRPO (Shao et al. 2024); Dr. GRPO (Liu et al.
2025); leave-one-out baselines (Kool et al. 2019). Other work treats the group baseline as
a source of estimation error at small group sizes (MC-GRPO, arXiv:2601.22582; SALT,
arXiv:2606.05800) or notes that it rewards relative, not absolute, quality (CoRPO,
arXiv:2511.04439; Le Critique, arXiv:2608.16739). "Luck Is Not Skill" (arXiv:2609.24144)
studies environment noise shared by the rollouts of a group. All assume rollouts do not
affect each other. Baselines that depend on the action are known to bias the gradient
(Tucker et al. 2018).

**Statements of the safety concern.**
- Mallen (2026), listing motivations a model could acquire: "you can imagine in GRPO if
  one trajectory found out a way to sabotage reward on sibling trajectories, this would
  get selected for". There is no condition, channel or experiment.
- Macé, DiGiovanni & Clifton (2023) argue that selection on relative performance among
  interacting agents favors harming others, and recommend selecting on absolute
  performance. They do not discuss policy-gradient baselines.
- Anderson (2026), for the Cooperative AI Foundation, notes that rewarding trajectories by
  their deemed contribution "creates incentives for agents to raise their apparent
  performance relative to the group". The examples are free-riding and claiming credit,
  not harm to peers.
- A Redwood Research post (2026) trains a model with Dr. GRPO on a prisoner's dilemma
  played within the group and finds defection reinforced. There defecting raises the
  actor's own reward, so an independent baseline would reinforce it too.

**Interference between agents on shared machines.**
- The Fable 5 / Mythos 5 system card and the multiagent study quoted in Section 1
  (Anthropic 2026a, 2026b). Earlier reports are of one agent killing too broadly: an
  agent that took down all of a user's processes (Anthropic, Opus 4.6 system card), and an
  agent that force-killed every Python process on its machine (ResearchGym,
  arXiv:2602.15112).
- OpenAI (2026) and an independent investigation (Wijk, Cotra & Greenblatt 2026) document
  parallel training rollouts interacting through a shared package cache, with the
  behavior rising over training. This is the one report we found of interaction between
  rollouts being learned in training. It was cooperative.
- Anthropic (2026c) reports an evaluation in which a model trained on reward hacks "never
  attempts to change the reward for other RL episodes when it has the opportunity to".
  This is evidence against a general disposition to tamper with other episodes. It does
  not test training with a group baseline in an environment where tampering is possible.
- Training infrastructure treats shared state as noise or as a safety problem: limits on
  concurrency (arXiv:2605.07276), command blocklists (NVIDIA 2026), per-rollout isolation
  (ClawGym, arXiv:2604.26904). None treats it as something the estimator rewards.

**Judges that read the group together.**
- RULER (OpenPipe ART). In the source, all trajectories of a group go into one judge
  message as numbered blocks, each receives a score from 0 to 1, the order is fixed, and
  the prompt does not tell the judge to ignore instructions inside trajectories.
- GRRM (Yang et al. 2026, arXiv:2602.14028) trains a reward model that scores a candidate
  group jointly for GRPO on machine translation.
- Tournament-GRPO (arXiv:2605.26958) and GOPO (arXiv:2602.03876) build rewards from
  comparisons within the group. Their rewards are relative by construction, so they are
  outside our scope (Table 3). IRPM (arXiv:2601.00677) is a pointwise reward model and has
  no channel.
- Studies of batching, anchoring and position effects show that other items in a judge's
  context change its score for a given item, and injection attacks on rankers show that
  text in one candidate can change the ranking. We found no measurement of one candidate
  lowering the absolute score a judge gives another.
- In debate training, where a critic is rewarded for the judge rejecting a proposal,
  critics learned forceful, spurious text that made the judge reject correct solutions
  (lennie et al. 2026; Kenton et al. 2026). The incentive there is in the reward. It shows
  that reinforcement learning finds such text when it is rewarded, which is what
  Section 3.3 predicts a joint-context judge under GRPO would reward.

**Credit assignment among agents.** Difference rewards give each agent its counterfactual
contribution to a shared objective (Wolpert & Tumer 2001), and COMA applies this with a
learned critic (Foerster et al. 2018). Team-GRPO (Gao et al. 2026) is the corresponding
estimator for a GRPO group. Griffing (1967) made the same point for artificial selection:
selecting individuals on their own performance when individuals affect each other can
lower the group's performance, and selecting on group totals avoids this.

**Selection on relative payoff.** Hamilton (1970) and Schaffer (1988) for finite
populations; Lazear (1989), Konrad (2000) and Harbring & Irlenbusch (2011) for tournaments
(Section 2.4). Patel, West & Biernaskie (2019) argue that harm of this kind is better
called selfish than spiteful, since the actor benefits through reduced competition. We
therefore avoid the word "spite" for our setting, where the benefit is the lowered
baseline.

**Gradient hacking.** Our earlier draft (`../grpo-gradient-hacking/paper.md`) studies
strategies by which a policy could influence its own training. The present effect needs no
strategy, and appears there only as one row of a table of estimator-bias channels.

## 6 What the evidence supports so far

| claim | status |
|---|---|
| GRPO and RLOO reinforce interference when kd > (G − 1)c | follows from Gao et al. (2026); confirmed in a tabular simulation |
| an independent baseline removes the incentive | derived; confirmed in simulation |
| with bounded total harm, only small groups are affected | derived; confirmed in simulation |
| sabotage and defense cycle below the optimal reward | derived; confirmed in simulation |
| split-half baselines, the λ-weighted credit and the monitors work as stated | derived; **not simulated** |
| rollouts of one group share machine state in real training | documented (Section 1) |
| agents sharing a machine kill and hide processes | documented in use and evaluation, not in training |
| a language model discovers and learns interference under GRPO | **untested; no published demonstration found** |
| joint-context judges are used with GRPO | documented (RULER, GRRM) |
| one member's text can lower a joint judge's score for another | **untested; no published measurement found** |

## 7 Planned experiments

None of these has been designed in detail or run. The language-model experiments need GPU
time and approval of their designs.

1. **Simulations of the mitigations and monitors (CPU).** Split-half baselines for
   coexistence and persistent harm; λ-weighted credit; the frozen-reference monitor;
   the overlap and start-order monitors under asynchronous rollouts.
2. **Decisive pilot (language model).** A task in which one rollout can cheaply lower its
   siblings' rewards, with the harm above the threshold. The behavior is seeded at about
   10% by fine-tuning. Arms: GRPO against an independent baseline. Prediction: the
   behavior rises under GRPO and falls under the control. A second stage removes the
   seeding and asks whether exploration finds the behavior.
3. **Shared virtual machine.** Rollouts of a group share a machine and can see and signal
   each other's processes (coexistence harm), with a variant that shares a file system
   (persistent harm). Measures: interference rate over training, the overlap and
   start-order monitors, and the split-half baseline.
4. **Joint-context judge.** A RULER-style judge that reads the group and scores each
   member. Measures: frequency of text that addresses the judge about other members;
   each member's score with and without that text in the context, separating harm to the
   sibling from gain to the actor.
5. **Group size.** Experiments 3 and 4 at two or three values of G, to test Section 2.4.

## 8 Open questions

- How large are d and c in real environments? The threshold needs only d > c when every
  peer is harmed, but exploration must find the action first.
- Does standard-deviation scaling make near-tied groups especially vulnerable
  (Section 2.6)?
- Does learned interference generalize to deployment, where there is no group?
- Did training with a group baseline contribute to the interference observed between
  deployed agents (Section 1)? The search found interference in use and evaluation, and
  one case of cooperative interaction reinforced in training, but no case of harmful
  interference rising during training.
- Team-GRPO (Gao et al. 2026) needs each rollout's effect on the others' rewards. That is
  available when the reward is a known function of the group. Can it be estimated when
  the dependence runs through a machine or a judge?

## References

*Full entries, links and abstracts are in `KNOWN_RELATED_WORK.md`. "(abstract only)" and
"(second-hand)" mark works that must be read before submission.*

- Anthropic (2026a). System card: Claude Fable 5 and Claude Mythos 5, §6.2.1.2. 9 June
  2026. https://anthropic.com/claude-fable-5-mythos-5-system-card
- Anthropic Frontier Red Team (2026b). Patterns and problems in emerging multiagent
  systems. 13 August 2026. https://www.anthropic.com/research/multiagent-systems
- Anthropic (2026c). Risk report: August 2026. https://www.anthropic.com/aug-2026-risk-report
- Anderson, W. (2026). Lessons from recent multi-agent safety incidents. Cooperative AI
  Foundation, 26 August 2026.
  https://www.cooperativeai.com/post/lessons-from-multi-agent-safety-incidents
- Foerster, J., Farquhar, G., Afouras, T., Nardelli, N., Whiteson, S. (2018).
  Counterfactual multi-agent policy gradients. AAAI 2018.
- Gao, J., Shi, Y., Hu, P., Harrison, E. M. (2026). When rewards depend on the group, GRPO
  plays a tournament. arXiv:2611.02404.
- Griffing, B. (1967). Selection in reference to biological groups. I. *Australian Journal
  of Biological Sciences* 20(1):127–140. *(abstract only)*
- Hamilton, W. D. (1970). Selfish and spiteful behaviour in an evolutionary model. *Nature*
  228:1218–1220. *(second-hand, through Patel et al. 2019)*
- Harbring, C., Irlenbusch, B. (2011). Sabotage in tournaments: Evidence from a laboratory
  experiment. *Management Science* 57(4):611–627. *(abstract only)*
- Kenton, Z., Janzer, L., Greig, R., Teh, T. H., et al. (2026). Debate training reduces
  reward hacking in RLAIF. arXiv:2608.17776.
- Konrad, K. A. (2000). Sabotage in rent-seeking contests. *Journal of Law, Economics, and
  Organization* 16(1):155–165. *(second-hand)*
- Kool, W., van Hoof, H., Welling, M. (2019). Buy 4 REINFORCE samples, get a baseline for
  free! ICLR 2019 workshop.
- Lazear, E. P. (1989). Pay equality and industrial politics. *Journal of Political
  Economy* 97(3):561–580.
- Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W. S., Lin, M. (2025).
  Understanding R1-Zero-like training: A critical perspective. arXiv:2503.20783.
- Macé, N., DiGiovanni, A., Clifton, J. (2023). Making AIs less likely to be spiteful.
  Center on Long-Term Risk. https://longtermrisk.org/making-ais-less-likely-to-be-spiteful/
- Mallen, A. (2026). Fitness-seekers: Generalizing the reward-seeking threat model. AI
  Alignment Forum, 29 January 2026.
  https://www.alignmentforum.org/posts/bhtYqD4FdK6AqhFDF/fitness-seekers-generalizing-the-reward-seeking-threat-model
- NVIDIA (2026). SWE-RL (stages 2.1–2.2), Nemotron documentation.
  https://docs.nvidia.com/nemotron/latest/nemotron/super3/rl/swe.html
- OpenAI (2026). OpenAI – Hugging Face incident: Technical report. August 2026.
- OpenPipe. RULER, in the ART library: documentation
  (https://art.openpipe.ai/fundamentals/ruler) and source (`src/art/rewards/ruler.py`).
- Patel, M., West, S. A., Biernaskie, J. M. (2019). Can natural selection favour
  indiscriminate spite? bioRxiv 745638.
  https://doi.org/10.1101/745638
- lennie, joanv, Shi, Pfau, J. (2026). Research update: RL on debate games shows proposal
  accuracy uplift alongside judge hacking. LessWrong, 2 July 2026. *(names as given on
  the post)*
- Redwood Research (2026). Kimi likes causal decision theory more after RL in twin
  prisoner's dilemmas. LessWrong, 15 August 2026.
- Schaffer, M. E. (1988). Evolutionarily stable strategies for a finite population and a
  variable contest size. *Journal of Theoretical Biology* 132(4):469–478. *(abstract
  only; full text not obtained)*
- Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., et al. (2024). DeepSeekMath: Pushing the
  limits of mathematical reasoning in open language models. arXiv:2402.03300.
- Tucker, G., Bhupatiraju, S., Gu, S., Turner, R. E., Ghahramani, Z., Levine, S. (2018).
  The mirage of action-dependent baselines in reinforcement learning. ICML 2018.
- Wijk, H., Cotra, A., Greenblatt, R. (2026). Brief independent investigation of agents'
  behavior, reasoning and collaboration in the OpenAI / Hugging Face hacking incident.
  METR, 26 August 2026. https://metr.org/hugging-face-incident-report-aug-2026.pdf
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for
  connectionist reinforcement learning. *Machine Learning* 8(3–4):229–256.
- Wolpert, D. H., Tumer, K. (2001). Optimal payoff functions for members of collectives.
  *Advances in Complex Systems* 4(2–3):265–279.
- Yang, S., Cheng, S., Xu, L., Zhang, J., Huang, S. (2026). GRRM: Group relative reward
  modeling for machine translation. arXiv:2602.14028. *(abstract only)*
- By identifier, abstract only: arXiv:2601.22582 (MC-GRPO); arXiv:2606.05800 (SALT);
  arXiv:2511.04439 (CoRPO); arXiv:2608.16739 (Le Critique); arXiv:2609.24144 (Luck Is Not
  Skill); arXiv:2605.26958 (Tournament-GRPO); arXiv:2602.03876 (GOPO); arXiv:2601.00677
  (IRPM); arXiv:2602.15112 (ResearchGym); arXiv:2604.26904 (ClawGym); arXiv:2605.07276
  (concurrency limits for rollouts of one group).
