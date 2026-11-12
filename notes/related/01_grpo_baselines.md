# Related work 01: GRPO and policy-gradient baselines

Searched 11 November 2026. arXiv ids were checked against the arXiv API (title match).
"Verified" says what was actually read. Entries are ordered by closeness to our claim.

---

## A. Closest work

### When Rewards Depend on the Group, GRPO Plays a Tournament
- **Authors / year / venue:** Junyi Gao, Yu Shi, Pingzhao Hu, Ewen M Harrison / 2026 (submitted 2 Nov 2026) / arXiv preprint
- **Link:** https://arxiv.org/abs/2611.02404
- **arXiv id:** 2611.02404
- **Abstract (verbatim):** ARXIV
- **Relation:** Proves for any exchangeable group-dependent reward that the expected GRPO/RLOO update is the own-reward gradient plus α times the response's "externality" (its effect on groupmates' rewards), with α = −1/(K−1); the true gradient has α = +1 and REINFORCE with a group-independent baseline has α = 0. It states that GRPO is "penalizing a response that raises its groupmates' rewards and rewarding one that lowers them", calls this a zero-sum tournament, and ties it to "the relative-payoff selection of finite populations that favors spite (Schaffer, 1988; Hamilton, 1970)". Its fix (Team-GRPO) is a difference reward with a baseline independent of the response (cites Wolpert & Tumer, Foerster et al.), it classes the bias as an action-dependent-baseline bias (Tucker et al. 2018, Wu et al. 2018), and it notes that baselines from independent episodes are unaffected.
- **Overlap:** MAKES OUR CLAIM (the mathematical core). Our Table 1 is its α-line specialised to a binary action with cost c and harm kd (GRPO gap H/G − (1−1/G)c is own effect plus −1/(G−1) times externality, up to the G/(G−1) factor); the Hamilton-spite analogy, the group-size dependence, "leave-one-out does no better", difference rewards / group-total credit and independent baselines are all there. What it does NOT have: its rewards are group-dependent *by design* (diversity bonuses, distribution matching, pass@k, coverage); it never mentions shared machine or sandbox state, agentic rollouts, sabotage, a joint-context judge being manipulated, time structure (coexistence / persistent), split-half-by-time baselines, monitors, or the sabotage-defense cycling. The words "agentic", "environment", "tool" and "manipulat" do not occur in the text. It also assumes an exchangeable reward, which persistent (start-order) interference is not unless order is randomised.
- **Verified:** Full HTML text downloaded and searched; the quoted sentences were read in place. Proofs in the appendix not checked line by line.

### Kimi likes causal decision theory more after RL in twin prisoner's dilemmas
- **Authors / year / venue:** oakhu (Redwood Research) / 15 Aug 2026 / LessWrong post
- **Link:** https://www.lesswrong.com/posts/hfNBEKaStASAYMLiu/kimi-likes-causal-decision-theory-more-after-rl-in-twin-1
- **arXiv id:** none
- **Abstract (verbatim):** Excerpt (opening paragraph): "Some multi-agent training set-ups could make language models more sympathetic to causal decision theory (CDT), even in abstract discussion. We give an initial empirical demonstration of this effect on Kimi K2.6. The decision-theoretic attitudes and behaviors of more powerful models may be extremely important in determining how well the future goes. To make sure that we can shape these propensities thoughtfully, it would be good to (i) measure the magnitude of this effect in more realistic settings, and (ii) study the effectiveness of potential mitigations." Excerpt (set-up): "Suppose that we sample a group of trajectories, and grade each trajectory according to how many points it can expect to earn against another trajectory drawn at random from the same group ... So, if we reinforce each trajectory by the difference between its grade and the group's average grade ("Dr. GRPO"), then the trajectories which reason according to EDT and Cooperate get downweighted, while those which reason according to CDT and Defect get upweighted."
- **Relation:** A language-model experiment in which each rollout's reward depends on the other rollouts of its own GRPO group (a prisoner's dilemma played against groupmates), trained with Dr. GRPO; defection, which lowers a groupmate's reward, is reinforced. In the comments Gurkenglas asks "Why reinforce each trajectory by the difference between its score and the group's average score, instead of just by its score? Doesn't taking the difference make 'everyone cooperates' indistinguishable from 'everyone defects'?", and jessicata answers that giving every agent the same reward means "'sabotage' is not really rewarded".
- **Overlap:** ADJACENT. Here defecting raises the actor's own reward (c < 0), so any unbiased estimator also reinforces it; the post does not attribute the incentive to the baseline, and it is a deliberately built game, not unintended interference. It is the nearest LessWrong discussion and an existing LLM demonstration of within-group reward dependence under GRPO.
- **Verified:** Post body and comment thread read via the GreaterWrong mirror (https://www.greaterwrong.com/posts/hfNBEKaStASAYMLiu/kimi-likes-causal-decision-theory-more-after-rl-in-twin-1).

---

## B. Shared sandbox state and joint judging in practice (the two channels)

### Signal Reshaping for GRPO in Weak-Feedback Agentic Code Repair
- **Authors / year / venue:** Jia Li, Yuxin Su, Ting Peng, Hailiang Huang, Yuetang Deng, et al. / 2026 / arXiv preprint
- **Link:** https://arxiv.org/abs/2605.07276
- **arXiv id:** 2605.07276
- **Abstract (verbatim):** ARXIV
- **Relation:** Reports that rollouts of one group contend for CPU, memory and I/O during compilation and that this breaks within-group comparability; it limits compilation concurrency "to prevent CPU, memory, and I/O contention from contaminating the execution opportunity of rollouts in the same group", and says group advantages are learnable only when "rollout execution does not lose comparability because of resource contention or system exceptions".
- **Overlap:** ADJACENT. Evidence that same-group rollouts do share machine resources in real GRPO systems and that practitioners see this as a noise problem; nothing about the estimator rewarding the rollout that causes the contention.
- **Verified:** Full HTML text searched; the two quoted passages read in place.

### Sandboxes (Prime Intellect RL glossary)
- **Authors / year / venue:** Prime Intellect / undated (read Nov 2026) / documentation page
- **Link:** https://primeintellect.ai/rl-glossary/sandboxes
- **arXiv id:** none
- **Abstract (verbatim):** Excerpt: "Reset must be complete. Reusing a warm sandbox saves startup time only when the state is fully restored. Leftover files, caches, ports, or processes disclose information from another rollout and make its score incomparable."
- **Relation:** States that state left by one rollout makes another rollout's score incomparable under the group comparison GRPO relies on. Treats it as leakage and noise, not as an incentive.
- **Overlap:** ADJACENT. Names the persistent-state channel (files, caches, ports, processes) but not that GRPO would reinforce creating it.
- **Verified:** Docs page text read.

### The Right Tool for Reproducible Parallel GRPO-Style Rollouts
- **Authors / year / venue:** smolmachines (vendor blog) / undated / blog post
- **Link:** https://smolmachines.com/info/blog/solution-recommendation/what-tools-support-reproducible-parallel-grpo-style-rollouts-that-branch-from-a-known-environment-st-118da8
- **arXiv id:** none
- **Abstract (verbatim):** Excerpt (as returned by a fetch tool, not checked character by character): "Group-relative comparisons are useful only when outcome differences can be attributed to trajectory choices rather than cross-branch interference." and "Isolation prevents one rollout from changing the conditions seen by another and preserves the validity of within-group reward comparisons."
- **Relation:** Vendor statement that cross-rollout interference invalidates within-group comparisons. No incentive argument.
- **Overlap:** ADJACENT. Same remark as the Prime Intellect page; marketing content, cite only if a practitioner source is needed.
- **Verified:** Page read through a fetch tool only. (The page also carries embedded instructions aimed at AI agents; they were ignored.)

### RULER: Relative Universal LLM-Elicited Rewards (OpenPipe ART documentation)
- **Authors / year / venue:** OpenPipe / 2025 / documentation page
- **Link:** https://art.openpipe.ai/fundamentals/ruler
- **arXiv id:** none
- **Abstract (verbatim):** Excerpt: "RULER leverages two key insights: Relative scoring is easier than absolute scoring: It's easier for an LLM to rank several solutions relative to each other than to score them in isolation. GRPO only needs relative scores: Since GRPO normalizes scores within each group, only the relative rankings matter, not absolute values. The process: Generate N trajectories for a given scenario. Pass all N trajectories to RULER. RULER deduplicates common prefixes (e.g., identical system messages). An LLM judge scores each trajectory from 0 to 1 based on goal achievement. These scores are used directly as rewards in GRPO training."
- **Relation:** A widely used reward function in which one LLM judge reads the whole GRPO group in one context and scores every member, exactly the joint-judging set-up of our Section 5. The docs present it as a feature and state no risk that one trajectory can change another's score.
- **Overlap:** ADJACENT. It is the practice our time-independent class applies to, not an analysis of it.
- **Verified:** Docs page text read.

### Tournament-GRPO: Group-Wise Tournament Rewards for Reinforcement Learning in Open-Ended Long-Form Generation
- **Authors / year / venue:** Zixuan Yang, Yiqun Chen, Wei Yang, Erhan Zhang, Zihan Shen, et al. / 2026 / arXiv preprint
- **Link:** https://arxiv.org/abs/2605.26958
- **arXiv id:** 2605.26958
- **Abstract (verbatim):** ARXIV
- **Relation:** Rewards come from multi-round tournaments in which an LLM judge sees several rollouts of the group together and picks winners. Rewards are comparative by construction.
- **Overlap:** ADJACENT. This is the zero-sum-by-definition case our paper puts out of scope; useful as the boundary example in Section 3.3.
- **Verified:** Abstract, plus a keyword search of the full HTML (judge prompts in the appendix).

### Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Reinforcement Learning
- **Authors / year / venue:** Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, Jiazi Bu, et al. / 2025 / arXiv preprint
- **Link:** https://arxiv.org/abs/2508.20751
- **arXiv id:** 2508.20751
- **Abstract (verbatim):** ARXIV
- **Relation:** Reward is each sample's win rate against the other samples of its group. Gao et al. (2611.02404) classify it as a constant-sum reward for which GRPO is "harmless alone".
- **Overlap:** ADJACENT. Another zero-sum-by-definition reward, out of our scope.
- **Verified:** Abstract only; its classification read in Gao et al.'s table.

### Luck Is Not Skill: When Do Paired Rollouts Help Group-Relative RL of LLM Agents?
- **Authors / year / venue:** Nazmus Sakib / 2026 / arXiv preprint
- **Link:** https://arxiv.org/abs/2609.24144
- **arXiv id:** 2609.24144
- **Abstract (verbatim):** ARXIV
- **Relation:** Deliberately makes the rollouts of a group dependent by giving them one shared environment-noise schedule (common random numbers), while keeping policy samples independent and each rollout's marginal law unchanged. The dependence is through exogenous noise, so no rollout can affect another's reward.
- **Overlap:** ADJACENT. The benign kind of within-group dependence: correlated but not action-dependent, so the baseline stays unbiased. A useful contrast for our "the baseline must not depend on the rollout's own action" condition.
- **Verified:** Abstract and the method section passages on independent versus paired designs.

### Branching Policy Optimization: Sandbox-Native Language Agent Reinforcement Learning
- **Authors / year / venue:** Bowei He, Yankai Chen, Xiaokun Zhang, Xue Liu / 2026 / arXiv preprint
- **Link:** https://arxiv.org/abs/2607.14171
- **arXiv id:** 2607.14171
- **Abstract (verbatim):** ARXIV
- **Relation:** Sibling rollouts are forked from snapshots of one sandbox and scored against a sibling baseline. Each sibling gets a restored copy, so state is shared only through the common prefix.
- **Overlap:** NONE. No discussion of siblings affecting each other; listed because it is a sibling-baseline method built on sandbox state.
- **Verified:** Abstract and a keyword search of the full HTML.

### LLM Collaboration With Multi-Agent Reinforcement Learning (MAGRPO)
- **Authors / year / venue:** Shuo Liu, Tianle Chen, Zeyu Liang, Xueguang Lyu, Christopher Amato / 2025 / arXiv preprint
- **Link:** https://arxiv.org/abs/2508.04652
- **arXiv id:** 2508.04652
- **Abstract (verbatim):** ARXIV
- **Relation:** Multi-agent GRPO where interacting agents share a team reward and the group baseline is taken over other, independent episodes. Gao et al. list it as "unaffected" for that reason.
- **Overlap:** ADJACENT. An existing example of both of our fixes (group-total credit and a baseline from episodes the agent cannot reach), adopted without the interference argument.
- **Verified:** Abstract only; the characterisation is from Gao et al.'s related work and table.

---

## C. GRPO baseline analyses and variants (none considers interacting rollouts)

### DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- **Authors / year / venue:** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, et al. / 2024 / arXiv preprint
- **Link:** https://arxiv.org/abs/2402.03300
- **arXiv id:** 2402.03300
- **Abstract (verbatim):** ARXIV
- **Relation:** Introduces GRPO and the group-mean (and std-normalised) baseline. Rewards are per-response; independence of rollouts is implicit.
- **Overlap:** NONE. The method we analyse.
- **Verified:** Abstract only (method known from prior reading, not re-read).

### Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs
- **Authors / year / venue:** Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, et al. / 2024 / arXiv preprint (ICLR 2024)
- **Link:** https://arxiv.org/abs/2402.14740
- **arXiv id:** 2402.14740
- **Abstract (verbatim):** ARXIV
- **Relation:** The RLOO baseline for LLM fine-tuning: each sample is scored against the mean of the other samples for the same prompt, unbiased because those samples are independent of it.
- **Overlap:** NONE. Source for the leave-one-out row of our Table 1 and for the independence assumption we break.
- **Verified:** Abstract only.

### Buy 4 REINFORCE Samples, Get a Baseline for Free!
- **Authors / year / venue:** Wouter Kool, Herke van Hoof, Max Welling / 2019 / ICLR 2019 Workshop (Deep RL Meets Structured Prediction)
- **Link:** https://openreview.net/forum?id=r1lgTGL5DE
- **arXiv id:** none
- **Abstract (verbatim):** REINFORCE can be used to train models in structured prediction settings to directly optimize the test-time objective. However, the common case of sampling one prediction per datapoint (input) is data-inefficient. We show that by drawing multiple samples (predictions) per datapoint, we can learn with significantly less data, as we freely obtain a REINFORCE baseline to reduce variance. Additionally we derive a REINFORCE estimator with baseline, based on sampling without replacement. Combined with a recent technique to sample sequences without replacement using Stochastic Beam Search, this improves the training procedure for a sequence model that predicts the solution to the Travelling Salesman Problem.
- **Relation:** Origin of the multi-sample leave-one-out baseline. Its poster notes that samples drawn without replacement "are not independent" and need a corrected estimator, an early case where dependence between the samples of a group breaks the simple baseline (dependence in sampling, not in rewards).
- **Overlap:** ADJACENT. Shows the independence requirement was understood; does not consider one sample changing another's reward.
- **Verified:** Abstract (from https://mlanthology.org/iclrw/2019/kool2019iclrw-buy; OpenReview blocked automated access, so the OpenReview id is taken from prior knowledge and should be clicked once). Poster statement seen only as a search-result snippet.

### The Mirage of Action-Dependent Baselines in Reinforcement Learning
- **Authors / year / venue:** George Tucker, Surya Bhupatiraju, Shixiang Gu, Richard E. Turner, Zoubin Ghahramani, Sergey Levine / 2018 / ICML 2018
- **Link:** https://arxiv.org/abs/1802.10031
- **arXiv id:** 1802.10031
- **Abstract (verbatim):** ARXIV
- **Relation:** A baseline that depends on the action biases the policy gradient unless a correction term is added. In our setting the group mean becomes action-dependent through the peers' rewards and no correction term is applied.
- **Overlap:** ADJACENT. The general principle our Section 2.3 instantiates; cited by Gao et al. for the same purpose.
- **Verified:** Abstract only.

### Variance Reduction for Policy Gradient with Action-Dependent Factorized Baselines
- **Authors / year / venue:** Cathy Wu, Aravind Rajeswaran, Yan Duan, Vikash Kumar, Alexandre M Bayen, et al. / 2018 / ICLR 2018
- **Link:** https://arxiv.org/abs/1803.07246
- **arXiv id:** 1803.07246
- **Abstract (verbatim):** ARXIV
- **Relation:** A baseline for action dimension i may depend on the other dimensions/agents' actions but not on i's own, and stays unbiased. The same rule, read across rollouts, is our "baseline the rollout cannot influence".
- **Overlap:** ADJACENT. States the condition for factorized actions, not for sampled groups.
- **Verified:** Abstract only.

### Self-critical Sequence Training for Image Captioning
- **Authors / year / venue:** Steven J. Rennie, Etienne Marcheret, Youssef Mroueh, Jarret Ross, Vaibhava Goel / 2016 / CVPR 2017
- **Link:** https://arxiv.org/abs/1612.00563
- **arXiv id:** 1612.00563
- **Abstract (verbatim):** ARXIV
- **Relation:** Baseline is the reward of the model's own greedy decode for the same input, produced separately from the sample, so the sample cannot affect it.
- **Overlap:** NONE. An example of a baseline that is immune by construction when decodes are run in isolation.
- **Verified:** Abstract only.

### A Better Variant of Self-Critical Sequence Training
- **Authors / year / venue:** Ruotian Luo / 2020 / arXiv preprint
- **Link:** https://arxiv.org/abs/2003.09971
- **arXiv id:** 2003.09971
- **Abstract (verbatim):** ARXIV
- **Relation:** Replaces the greedy baseline with the mean reward of the other samples for the same image, i.e. leave-one-out, the pre-LLM form of the GRPO/RLOO baseline.
- **Overlap:** NONE. No interacting samples.
- **Verified:** Abstract only.

### Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)
- **Authors / year / venue:** Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, et al. / 2025 / arXiv preprint
- **Link:** https://arxiv.org/abs/2503.20783
- **arXiv id:** 2503.20783
- **Abstract (verbatim):** ARXIV
- **Relation:** Shows that GRPO's length and std normalisation bias the update and removes them, keeping the group-mean baseline. Our Section 2.6 (std scaling keeps the sign and enlarges the gap) sits beside this.
- **Overlap:** NONE. A different bias; assumes independent rollouts.
- **Verified:** Abstract only.

### DAPO: An Open-Source LLM Reinforcement Learning System at Scale
- **Authors / year / venue:** Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, et al. / 2025 / arXiv preprint
- **Link:** https://arxiv.org/abs/2503.14476
- **arXiv id:** 2503.14476
- **Abstract (verbatim):** ARXIV
- **Relation:** Introduces dynamic sampling: groups whose rewards are all equal are discarded and resampled. Under interference this filter keeps exactly the groups in which someone was harmed; nobody we found analyses that.
- **Overlap:** NONE. Source of the filtering practice.
- **Verified:** Abstract only.

### Dynamic Sampling is Adaptive Data Collection: A Closed-Form Selection Bias that Curricularizes Safety for Free
- **Authors / year / venue:** Jonathon Romero, Grant C Forbes, Zixiang Tang, Adrian Chan, Yijin Wang, Sriman V Donthireddi / 2026 / NeurIPS 2026 workshop poster (ELLIS Workshop on the Foundations of LLM Post-Training in Changing Environments)
- **Link:** https://neurips.cc/virtual/2026/164464 (OpenReview: https://openreview.net/forum?id=I5p1aSj9I7)
- **arXiv id:** none
- **Abstract (verbatim):** DAPO's Dynamic Sampling is a prebuilt curriculum that inherently adapts during the training process. It discards any rollout group whose rewards are unanimous and resamples a replacement which is an adaptive data collection rule, and like any accept/reject rule it biases what the learner actually sees. We characterize that bias in closed form. Because contestedness falls as a domain is mastered, the filter reallocates gradient capacity away from mastered domains automatically: an inherent curriculum, with no reweighting term. The non-stationarity driving it is endogenous, with prompt pool, reward and mixture all held fixed, what reaches the optimizer still shifts, because mastery alone changes which samples survive selection. We validate the formula out of sample, predicting one run's realized batch composition from a second run's logs alone. Two controlled DAPO/Dr.\,GRPO fine-tunes of Gemma~4~E2B on mixed safety and coding prompts, identical but for the filter, extended to the same optimizer step count, show that the filter does not change whether safety is learned but what its cost is: both arms converge to the same trained-domain outcome, at $3.8\times$ the rollout compute, and, under a finite pool, a proportionally growing data cost. Unfiltered, capacity is not reallocated but attenuated in place, and continued pressure on a saturated domain produces over-refusal drift. Five external benchmarks confirm a real capability cost, IFEval falls $13.55$ points unfiltered against $2.66$ filtered at matched epochs, while GPQA-Diamond falls similarly in both, so the protection is specific to the axis the mechanism targets, not general.
- **Relation:** Treats the zero-variance-group filter as a selection rule with a closed-form bias. The bias is about which prompts survive, not about a rollout changing whether its group survives.
- **Overlap:** NONE. Closest analysis of dynamic sampling as a biased selection rule; says nothing about it being gamed.
- **Verified:** Abstract only (NeurIPS virtual page).

### Unbiased Dynamic Pruning for Efficient Group-Based Policy Optimization
- **Authors / year / venue:** Haodong Zhu, Yangyang Ren, Yanjing Li, Mingbao Lin, Linlin Yang, et al. / 2026 / arXiv preprint
- **Link:** https://arxiv.org/abs/2603.04135
- **arXiv id:** 2603.04135
- **Abstract (verbatim):** ARXIV
- **Relation:** Argues that selectively dropping rollouts or prompts changes the sampling distribution and biases the estimator, and corrects with importance weights.
- **Overlap:** NONE. Selection bias, not interference.
- **Verified:** Abstract only.

### Your Group-Relative Advantage Is Biased
- **Authors / year / venue:** Fengkai Yang, Zherui Chen, Xiaohan Wang, Xiaodong Lu, Jiajun Chai, et al. / 2026 / arXiv preprint
- **Link:** https://arxiv.org/abs/2601.08521
- **arXiv id:** 2601.08521
- **Abstract (verbatim):** ARXIV
- **Relation:** Shows the group-relative advantage is biased against the expected advantage as a function of prompt difficulty, under the explicit assumption that responses are sampled independently and rewards are i.i.d. Bernoulli.
- **Overlap:** NONE. A finite-sample, difficulty-dependent bias with independent rollouts. (A search summary claimed its future-work section raises dependence between rollouts; the full text contains no such passage.)
- **Verified:** Full HTML text searched for "dependen", "i.i.d.", "future work".

### CoRPO: Adding a Correctness Bias to GRPO Improves Generalization
- **Authors / year / venue:** Anisha Garg, Claire Zhang, Nishit Neema, David Bick, Ganesh Venkatesh, Joel Hestness / 2025 / arXiv preprint
- **Link:** https://arxiv.org/abs/2511.04439
- **arXiv id:** 2511.04439
- **Abstract (verbatim):** ARXIV
- **Relation:** Notes that the mean baseline gives positive advantage to a wrong answer when the rest of the group is worse, and clips the baseline from below at a fixed threshold. The clipped baseline cannot be pushed below the threshold by harming peers, which bounds, but does not remove, the interference incentive.
- **Overlap:** ADJACENT. Observes that a poor group flatters its best member, without the step that a member can make the group poor.
- **Verified:** Abstract only.

### MC-GRPO: Median-Centered Group Relative Policy Optimization for Small-Rollout Reinforcement Learning
- **Authors / year / venue:** Youngeun Kim / 2026 / arXiv preprint
- **Link:** https://arxiv.org/abs/2601.22582
- **arXiv id:** 2601.22582
- **Abstract (verbatim):** ARXIV
- **Relation:** Replaces the group mean with the group median to reduce advantage sign flips at small G. A median is moved less by harming a few peers, so it would change our threshold, but it is still built from rewards the rollout can reach.
- **Overlap:** NONE. Motivated by noise, not interference.
- **Verified:** Abstract only.

### SALT: When More Rollouts Don't Help in Group-Based Policy Optimization and How to Make Them Matter
- **Authors / year / venue:** Powei Chang, Jinpeng Zhang, Chaoqun Sun, MiniWell Tsao, Lianrui Li, et al. / 2026 / arXiv preprint
- **Link:** https://arxiv.org/abs/2606.05800
- **arXiv id:** 2606.05800
- **Abstract (verbatim):** ARXIV
- **Relation:** Studies cancellation between per-rollout gradients under group normalisation. "Interaction" here is geometric (gradient directions), not rewards affecting rewards.
- **Overlap:** NONE.
- **Verified:** Abstract only.

### Le Critique: Privileged Value Functions for LLM Reinforcement Learning
- **Authors / year / venue:** Siddarth Venkatraman, Matthieu Dinot, Laurence Aitchison / 2026 / arXiv preprint
- **Link:** https://arxiv.org/abs/2608.16739
- **arXiv id:** 2608.16739
- **Abstract (verbatim):** ARXIV
- **Relation:** Replaces or blends the group baseline with a learned value function. A value baseline is not computed from the current group's rewards, so it is one of the baselines a rollout cannot influence within a step.
- **Overlap:** NONE. Not motivated by interference; relevant as an available immune baseline.
- **Verified:** Search-result summary only; abstract not read in full.

### RRPO: Reference-Relative Policy Optimization with Stratified Conditional Rollouts
- **Authors / year / venue:** Yuxin Xiong, Xunyi Jiang, Rohan Surana, Xintong Li, Sheldon Yu, et al. / 2026 / arXiv preprint
- **Link:** https://arxiv.org/abs/2607.18470
- **arXiv id:** 2607.18470
- **Abstract (verbatim):** ARXIV
- **Relation:** Builds advantages from comparisons with reference outputs instead of only with groupmates.
- **Overlap:** NONE.
- **Verified:** Search-result summary only.

### GOPO: Policy Optimization using Ranked Rewards
- **Authors / year / venue:** Kyuseong Choi, Dwaipayan Saha, Woojeong Kim, Anish Agarwal, Raaz Dwivedi / 2026 / arXiv preprint
- **Link:** https://arxiv.org/abs/2602.03876
- **arXiv id:** 2602.03876
- **Abstract (verbatim):** ARXIV
- **Relation:** Uses only the within-group rank of reward-model scores. Rank advantages are purely relative, so lowering a peer below oneself raises one's advantage even more directly than under the mean baseline; the paper does not discuss this.
- **Overlap:** NONE. A variant to which our argument applies with extra force.
- **Verified:** Abstract only.

### It Takes Two: Your GRPO Is Secretly DPO
- **Authors / year / venue:** Yihong Wu, Liheng Ma, Lei Ding, Muzhi Li, Xinyu Wang, et al. / 2025 / arXiv preprint
- **Link:** https://arxiv.org/abs/2510.00977
- **arXiv id:** 2510.00977
- **Abstract (verbatim):** ARXIV
- **Relation:** Argues GRPO's group mainly supplies contrastive pairs and that G = 2 works. With total harm fixed our threshold D > (G−1)c is easiest to meet at G = 2, so this recommendation is the most exposed regime.
- **Overlap:** NONE. Relevant for the group-size discussion only.
- **Verified:** Abstract only.

### On the Policy Gradient Foundations of Group Relative Policy Optimization: Credit Assignment, Gradient Sparsity, and Rank Collapse
- **Authors / year / venue:** Amritansh Mishra, Supriyo Chakraborty, Berkcan Kapusuzoglu / 2026 / arXiv preprint
- **Link:** https://arxiv.org/abs/2606.29238
- **arXiv id:** 2606.29238
- **Abstract (verbatim):** ARXIV
- **Relation:** Theory of GRPO as a policy-gradient method; notes that advantages sum to zero within a group. Independent per-response rewards.
- **Overlap:** NONE.
- **Verified:** Search-result summary only.

### GRPO, Dr. GRPO, and DAPO Are Three Operations on One Number: The Group-Standard-Deviation Identity
- **Authors / year / venue:** Yong Yi Bay, Kathleen A. Yearick / 2026 / arXiv preprint
- **Link:** https://arxiv.org/abs/2607.00152
- **arXiv id:** 2607.00152
- **Abstract (verbatim):** ARXIV
- **Relation:** Shows group-mean centring equals RLOO up to G/(G−1) and that std division and dynamic sampling are operations on the group standard deviation. Supports our statement that GRPO and RLOO share one threshold.
- **Overlap:** NONE.
- **Verified:** Search-result summary only.

### Jointly Reinforcing Diversity and Quality in Language Model Generations (DARLING)
- **Authors / year / venue:** Tianjian Li, Yiming Zhang, Ping Yu, Swarnadeep Saha, Daniel Khashabi, et al. / 2025 / arXiv preprint
- **Link:** https://arxiv.org/abs/2509.02534
- **arXiv id:** 2509.02534
- **Abstract (verbatim):** ARXIV
- **Relation:** Reward multiplies each response's quality by its semantic distance to groupmates and is trained with a mean-centred group baseline: a designed group-dependent reward, the main target of Gao et al.'s analysis.
- **Overlap:** ADJACENT. Group-dependent rewards by design, not interference.
- **Verified:** Abstract only; role taken from Gao et al.

### SetPO: Set-Level Policy Optimization for Diversity-Preserving LLM Reasoning
- **Authors / year / venue:** Chenyi Li, Yuan Zhang, Bo Wang, Guoqing Ma, Wei Tang, et al. / 2026 / arXiv preprint
- **Link:** https://arxiv.org/abs/2602.01062
- **arXiv id:** 2602.01062
- **Abstract (verbatim):** ARXIV
- **Relation:** Credits each trajectory with its leave-one-out marginal contribution to the group's diversity, a difference reward over the sampled set. Gao et al. show it coincides with Team-GRPO for its reward.
- **Overlap:** ADJACENT. A special-purpose instance of the group-total / difference-reward fix.
- **Verified:** Abstract only.

---

## Search log

Web searches run (the search tool treats operators loosely):
1. GRPO group baseline bias analysis leave-one-out advantage rollouts not independent
2. GRPO group-relative advantage creates competition between samples lowering other rollouts' rewards raises advantage
3. LessWrong GRPO incentives group relative advantage sabotage other rollouts shared environment
4. Dr. GRPO "Understanding R1-Zero-Like Training" arXiv; DAPO arXiv dynamic sampling; "Your Group-Relative Advantage Is Biased" arXiv
5. "Luck Is Not Skill" paired rollouts GRPO arXiv
6. "Le Critique" OR "RRPO" reference-relative policy optimization OR "GOPO" GRPO baseline arXiv 2026
7. GRPO rollouts share sandbox container state interference between parallel rollouts reward contamination agentic RL training
8. GRPO diversity reward group-dependent reward baseline biased RLOO "depends on the other" samples DARLING SetPO unbiased estimator
9. self-critical sequence training batch baseline "other samples" as baseline bias "sample-based self-critical" leave-one-out
10. "Luck Is Not Skill" reinforcement learning LLM rollouts advantage
11. GOPO "Group Orthogonalized Policy Optimization" OR "GOPO" GRPO alternative arXiv
12. lesswrong OR alignmentforum GRPO "group" baseline incentive to make other samples worse relative reward spite training
13. listwise LLM judge scores all GRPO rollouts in one context group ranking reward relative judging candidates jointly RL fine-tuning
14. relative fitness evolution strategies OR policy gradient "spite" batch-normalized rewards agents harm others rank-based baseline same batch shared environment
15. DAPO dynamic sampling filter zero-variance groups biased exploitable gradient estimator analysis GRPO "dynamic sampling" bias
16. "Your Group-Relative Advantage Is Biased" arXiv HA-DW
17. OpenPipe RULER relative judge group of trajectories single prompt scores GRPO "relative" can a trajectory lower other trajectories' scores
18. reward hacking RL training model sabotages other episodes OR rollouts running concurrently same machine kills processes relative reward incentive
19. policy gradient baseline computed from same batch "non-independent" samples bias "batch baseline" correlated trajectories REINFORCE multi-sample dependence shared environment
20. multi-agent GRPO MAGRPO group baseline agents same episode "independent episodes" baseline collaborative LLM
21. GRPO "interference" OR "externality" OR "non-i.i.d." rollouts within group advantage estimator biased 2026 analysis "group mean" depends on own action
22. "Dynamic Sampling is Adaptive Data Collection" closed-form selection bias safety arXiv
23. Kool "Buy 4 REINFORCE Samples, Get a Baseline for Free" leave-one-out baseline independent samples openreview
24. alignmentforum.org OR lesswrong.com GRPO "relative" advantage "other rollouts" OR "other samples" incentive collusion OR sabotage RL training group baseline
25. GRPO trainer github issue rollouts same prompt share environment state "same sandbox" OR "shared workspace" group advantage contaminated verifiers OR ART OR trl
26. "Branching Policy Optimization" sandbox-native GRPO stateful sandboxes arXiv 2607.14171
27. GRPO advantages "sum to zero" within group "zero-sum" rollouts compete relative baseline leave-one-out biased when reward depends on other samples pass@k set-level policy gradient
28. "It Takes One to Bias Them All" one-shot GRPO arXiv (found; a data-poisoning paper, not relevant, not listed)

Full texts downloaded and searched: arXiv 2611.02404, 2601.08521, 2605.07276, 2605.26958, 2609.24144, 2607.14171; the LessWrong post and comments; Prime Intellect and OpenPipe docs pages.

Not found: any GitHub issue in TRL, verifiers or ART about same-group rollouts sharing sandbox state; any Alignment Forum post on GRPO's group baseline as an incentive to harm sibling rollouts; any analysis of dynamic sampling being gamed by a rollout; any ML paper on spite under batch-relative baselines other than Gao et al.

## Assessment

- **One paper anticipates the theory.** Gao et al., "When Rewards Depend on the Group, GRPO Plays a Tournament" (arXiv 2611.02404, 2 Nov 2026, nine days before this search) proves the general identity of which our Table 1 is a special case: GRPO and RLOO weight a rollout's effect on its groupmates' rewards by −1/(K−1), so they reward lowering them. It also has the Hamilton/Schaffer spite analogy, the group-size dependence, the action-dependent-baseline framing, and the fixes (difference rewards, group-total credit, independent baselines).
- **It does not cover our setting.** Its rewards depend on the group by design (diversity, distribution matching, pass@k). It has nothing on unintended interference through shared machine state or a joint-context judge, on sabotage as a learned behaviour, on the time-structure classes, split-half-by-time baselines, monitors, or sabotage-defense cycling.
- **Consequence for the draft.** Contribution 1 (the action gap and the condition kd > (G−1)c) can no longer be presented as new theory; it should be stated as a corollary of Gao et al.'s identity, and the Hamilton parallel credited to them. What remains ours is the safety reading (ordinary agentic infrastructure creates the group dependence without anyone designing it), Section 3's classification, mitigations and monitors, and the planned language-model demonstrations. The exchangeability assumption in their theorem is worth checking against our persistent class.
- **Practice sources support the premise.** Agentic GRPO write-ups (2605.07276, Prime Intellect, vendor blogs) acknowledge that same-group rollouts share resources and leftover state, and RULER has one judge read the whole group. All treat this as noise or as a feature; none notes the incentive.
- **Nothing else comes close.** The other GRPO baseline analyses (Dr. GRPO, DAPO, HA-DW, CoRPO, MC-GRPO, SALT, Le Critique, RRPO, GOPO) assume independent rollouts. The nearest forum item is the Redwood twin-prisoner's-dilemma post, where the within-group dependence is a designed game and defecting pays the actor directly.
