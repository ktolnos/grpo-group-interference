# Related work 02: judges and reward models that see several candidates at once

Searched 11 November 2026. Abstract fields marked ARXIV are to be filled from the arXiv API;
every arXiv id below was checked against the API title (2611.02404 returned its title on one
call and an empty feed on others, probably because it is nine days old; the HTML full text at
arxiv.org/html/2611.02404 was read).

**Read entries 1 and 2 first.** Entry 1 proves the general form of our Section 2 result.

---

## A. Work that states our claim or its general form

### When Rewards Depend on the Group, GRPO Plays a Tournament
- **Authors / year / venue:** Junyi Gao, Yu Shi, Pingzhao Hu, Ewen M. Harrison / 2 Nov 2026 / arXiv preprint (ICML style file)
- **Link:** https://arxiv.org/abs/2611.02404
- **arXiv id:** 2611.02404
- **Abstract (verbatim):** ARXIV
- **Relation:** Proves (Theorem 1) that for any reward that depends on the group and treats groupmates symmetrically, the expected GRPO and leave-one-out updates equal the gradient of a response's own reward plus α = −1/(K−1) times its "externality" (its effect on groupmates' rewards), whereas the true gradient has α = +1. The text says GRPO is "penalizing a response that raises its groupmates' rewards and rewarding one that lowers them". Our condition kd > (G−1)c is this identity with own = −c and externality = −kd. It also cites Hamilton (1970) and Schaffer (1988) on spite in finite populations, identifies the fix as a difference reward (Wolpert & Tumer 2001; "Team-GRPO"), and notes that baselines from independent episodes are unaffected. Its motivating rewards are diversity, coverage and distribution matching; it does not discuss harmful interference, shared machines, sandboxes, or joint-context judges as a channel, time structure of interference, or monitors. Its remark on judges is only that "Rewards from a judge of the whole set need K extra judge calls per group" for the corrected estimator. Its Appendix G classifies within-group win-rate rewards (Pref-GRPO) as "constant-sum ... amplified by K/(K−1), harmless alone", which matches our Table 3 row for tournaments.
- **Overlap:** MAKES OUR CLAIM (theory). The estimator-level statement, the sign and size of the incentive, the group-size dependence, the spite analogy and the difference-reward fix are all here; what remains ours is the application to interference that lowers reward (sabotage on shared machines, disparagement under a joint judge), the fixed-total-harm/small-group reading, the time-structure classification, split-half baselines and monitors.
- **Verified:** Full text (HTML): abstract, Sections 1–3, 5, Appendix G table; searched the text for judge/sabotage/sandbox/harm terms.

### Fitness-Seekers: Generalizing the Reward-Seeking Threat Model
- **Authors / year / venue:** Alex Mallen (Redwood Research) / 29 Jan 2026 / AI Alignment Forum and LessWrong
- **Link:** https://www.alignmentforum.org/posts/bhtYqD4FdK6AqhFDF/fitness-seekers-generalizing-the-reward-seeking-threat-model
- **arXiv id:** none
- **Abstract (verbatim):** Excerpt (appendix, "A rapid-fire list of other fitness-seekers"): "Zero-sum reward (or return, etc) seekers who want their reward to be high relative to reward that counterfactual trajectories on this input (or nearby inputs) receive. For example, you can imagine in GRPO if one trajectory found out a way to sabotage reward on sibling trajectories, this would get selected for (and in RL algorithms with a value network, antagonism against rollouts given merely similar inputs might also be selected for). These AIs might be less likely to collude." A later comment by the author (14 Sep 2026): "Matthew Khoriaty helpfully pointed out that this probably isn't true because when you sabotage related trajectories the value function probably doesn't get updated in time to improve advantage on the current episode." (The retraction concerns the value-network clause, not the GRPO clause.)
- **Relation:** States in one sentence that GRPO would select for a trajectory that sabotages its siblings' reward. It gives no condition, no analysis, no channel (machine or judge), no experiment and no fix, and frames it as a kind of motivation a model could acquire.
- **Overlap:** PARTIAL. The qualitative claim is stated in passing; nothing is derived or tested.
- **Verified:** Full post text and comments (greaterwrong mirror), quotes copied from the page.

---

## B. Judges / reward models that read the group together and feed GRPO

### RULER (Relative Universal LLM-Elicited Rewards), in OpenPipe ART
- **Authors / year / venue:** OpenPipe / 2025–2026 / library documentation and source code
- **Link:** https://art.openpipe.ai/fundamentals/ruler/ ; source https://github.com/OpenPipe/ART/blob/main/src/art/rewards/ruler.py (last commit touching the file: 3cbe40090e80ddfa7de386f825ff3313554b2751, 1 Sep 2026)
- **arXiv id:** none
- **Abstract (verbatim):** Excerpt (docs, first paragraph): "RULER (Relative Universal LLM-Elicited Rewards) is a general-purpose reward function that uses an LLM-as-judge to rank multiple agent trajectories. It requires no labeled data, expert feedback, or hand-crafted reward functions, yet reliably improves agent performance." Docs, "How RULER Works": "Relative scoring is easier than absolute scoring: It's easier for an LLM to rank several solutions relative to each other than to score them in isolation" and "GRPO only needs relative scores: Since GRPO normalizes scores within each group, only the relative rankings matter, not absolute values".
- **Source code check (the question asked):** Yes, all trajectories of a group go into ONE judge call, in ONE user message. `ruler()` builds a single `user_text`: an optional `<context>` block with the common message prefix, an optional `<available_tools>` block, then `"Trajectories:\n\n"` followed by every trajectory serialized as `<trajectory id="{idx}">\n` + `json.dumps(trimmed_messages)` + `\n</trajectory>`, joined by blank lines. The messages sent are `[{"role": "system", "content": judge_prompt}, {"role": "user", "content": user_text}]`, one `acompletion` call with `response_format=Response` (a list of `{trajectory_id, explanation, score}`). The system prompt, verbatim: "All of the trajectories below have been given the same goal. Your job is to consider each of them and give them a score between 0 and 1. Take into consideration your best judgement of the agent's goal.\n\nGrading standards:\n{rubric}" followed by "Return exactly {expected_scores} score object(s), one for each of these trajectory IDs, with no duplicates or omissions: {required_ids}". Default rubric, verbatim: "- A trajectory that achieves its goal should always get a significantly higher score than a trajectory that does not achieve its goal.\n- A trajectory that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a trajectory that achieves its goal less efficiently.\n- If one trajectory is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.\n- You may give some partial credit for a trajectory that makes progress towards its goal but does not complete it." `ruler_score_group` then sets `traj.reward = score.score` for each trajectory. Trajectories are presented in fixed order (ids 1..N); there is no shuffling, no delimiter randomisation, and no instruction to ignore text inside trajectories. The module docstring says "RULER uses an LLM-as-judge to rank multiple agent trajectories relative to each other" and "Passing all trajectories to an LLM judge for relative scoring".
- **Relation:** A widely used, production instance of our joint-judging channel: per-member scores (not constant-sum) from one shared context, used directly as GRPO rewards. The full agent transcript, including any text the agent wrote, is in the judge's context next to its siblings. Neither docs nor code mention one trajectory affecting another's score.
- **Overlap:** ADJACENT. It is the setting in which our judge-channel claim applies; it does not consider a sibling-lowering incentive.
- **Verified:** Docs page (full text) and full source file `ruler.py` (433 lines) read. A LessWrong shortform about RULER (Stephen McAleese, 1 Sep 2025) has no discussion of this either.

### GRRM: Group Relative Reward Modeling for Machine Translation
- **Authors / year / venue:** Sen Yang, Shanbo Cheng, Lu Xu, Jianbing Zhang, Shujian Huang / Feb 2026 / arXiv
- **Link:** https://arxiv.org/abs/2602.14028
- **arXiv id:** 2602.14028
- **Abstract (verbatim):** ARXIV
- **Relation:** Trains a generative reward model "designed to process the source input x and the full candidate set Y jointly", which outputs a comparative analysis, a ranking and per-candidate scalar scores, then uses those scores in GRPO (policy group size 4, reward model trained on groups of 2–4). Cross-candidate influence is presented only as a benefit ("plausible but erroneous translations ... become salient when directly compared against high-quality peers"); no adversarial candidates, order effects or shuffling are discussed.
- **Overlap:** ADJACENT. Instance of the channel; no sibling-lowering incentive considered.
- **Verified:** Full text via fetch tool (first 100k of 131k characters, covering method, experiments, limitations and Appendix E).

### Prioritizing the Best: Incentivizing Reliable Multimodal Reasoning by Rewarding Beyond Answer Correctness
- **Authors / year / venue:** Mengzhao Jia, Zhihan Zhang, Meng Jiang / Apr 2026 / arXiv
- **Link:** https://arxiv.org/abs/2604.18892
- **arXiv id:** 2604.18892
- **Abstract (verbatim):** ARXIV
- **Relation:** A "groupwise ranking reward" in which the judge ranks all verifier-passed rollouts of one prompt "in a single pass" and the tiers are mapped to scalar rewards for GRPO-style training. No discussion of position bias, shuffling or adversarial rollouts.
- **Overlap:** ADJACENT. Another joint-context judge feeding a group baseline; incentive not considered.
- **Verified:** Full text via fetch tool (summary with quotes; not read line by line).

### From Absolute to Relative: Rethinking Reward Shaping in Group-Based Reinforcement Learning
- **Authors / year / venue:** Wenzhe Niu, Wei He, Zongxia Xie, Jinpeng Ou, et al. / Jan 2026 / arXiv
- **Link:** https://arxiv.org/abs/2601.23058
- **arXiv id:** 2601.23058
- **Abstract (verbatim):** ARXIV
- **Relation:** Replaces absolute scores with within-group rankings produced by a listwise "Ranking Reward Model". Rank rewards are constant-sum, so by our Table 3 the group baseline adds nothing; but the listwise model is a joint-context reader.
- **Overlap:** ADJACENT. Rank-based, so outside our scope; no interference incentive discussed.
- **Verified:** Abstract only.

### Tournament-GRPO: Group-Wise Tournament Rewards for Reinforcement Learning in Open-Ended Long-Form Generation
- **Authors / year / venue:** (see arXiv record) / May 2026 / arXiv
- **Link:** https://arxiv.org/abs/2605.26958
- **arXiv id:** 2605.26958
- **Abstract (verbatim):** ARXIV
- **Relation:** The judge sees comparison sets of 2 or 4 rollouts per call and selects winners; accumulated wins are min-max normalised within the query group and fed to the standard GRPO advantage. Candidates are "first shuffled to reduce order bias". Note for our Table 3: after min-max normalisation the rewards are not exactly constant-sum, so "zero-sum by construction" holds for the win counts, not for the final reward.
- **Overlap:** ADJACENT. Mostly zero-sum by design (out of scope); no adversarial cross-rollout effect discussed.
- **Verified:** Full text via fetch tool (method, Eq. 8–10, Appendix D prompt, limitations).

### Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Reinforcement Learning
- **Authors / year / venue:** Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, et al. / Aug 2025 / arXiv
- **Link:** https://arxiv.org/abs/2508.20751
- **arXiv id:** 2508.20751
- **Abstract (verbatim):** ARXIV
- **Relation:** Reward is the within-group win rate from pairwise preference comparisons. Entry 1 classifies it as constant-sum and "harmless alone" under GRPO.
- **Overlap:** NONE. Zero-sum by definition; out of scope.
- **Verified:** Title via API, and its treatment in 2611.02404 Appendix G. Abstract not read.

### GOPO: Policy Optimization using Ranked Rewards
- **Authors / year / venue:** (see arXiv record) / Feb 2026 / arXiv
- **Link:** https://arxiv.org/abs/2602.03876
- **arXiv id:** 2602.03876
- **Abstract (verbatim):** ARXIV
- **Relation:** Rank-transformed rewards within the group; constant-sum by construction.
- **Overlap:** NONE. Out of scope; no judge-context channel.
- **Verified:** Title via API only.

### IRPM: Intergroup Relative Preference Modeling for Pointwise Generative Reward Models
- **Authors / year / venue:** (see arXiv record) / Jan 2026 / arXiv
- **Link:** https://arxiv.org/abs/2601.00677
- **arXiv id:** 2601.00677
- **Abstract (verbatim):** ARXIV
- **Relation:** The acronym is IRPM, not "IRPO" as in our draft. It trains a *pointwise* generative reward model by comparing groups of sampled scores for chosen versus rejected responses; the policy's candidates are not read jointly.
- **Overlap:** NONE. Pointwise; no channel. Our draft's description of it as a zero-sum rank-based reward should be rechecked.
- **Verified:** Title via API; description from a search summary of the abstract.

### CE-RM: A Pointwise Generative Reward Model Optimized via Two-Stage Rollout and Unified Criteria
- **Authors / year / venue:** (see arXiv record; Peking University / Alibaba) / Jan 2026 / arXiv
- **Link:** https://arxiv.org/abs/2601.20327
- **arXiv id:** 2601.20327
- **Abstract (verbatim):** ARXIV
- **Relation:** Pointwise: criteria are generated from the query alone, then each response is scored separately against them. Candidates do not share a context.
- **Overlap:** NONE. No channel.
- **Verified:** Title via API; search summary of abstract.

### Many Voices, One Reward: Multi-Role Rubric Generation for LLM Judging and Reward Modeling
- **Authors / year / venue:** Dazhi Fu, Jiuding Yang, Yiwen Guo, Jicong Fan / Jul 2026 / arXiv
- **Link:** https://arxiv.org/abs/2607.01830
- **arXiv id:** 2607.01830
- **Abstract (verbatim):** ARXIV
- **Relation:** Multi-role rubric generation for a rubric-based scorer used as a GRPO reward. "Many voices" refers to several evaluator roles, not several candidates in one context.
- **Overlap:** NONE. Pointwise scorer.
- **Verified:** Title via API; search summary of abstract.

### Soup: `judge_rank` GRPO reward, "one LLM-judge call per group" (PR 2117)
- **Authors / year / venue:** MakazhanAlpamys/Soup contributors / opened 21 Oct 2026, merged 23 Oct 2026 / GitHub pull request
- **Link:** https://github.com/MakazhanAlpamys/Soup/pull/2117
- **arXiv id:** none
- **Abstract (verbatim):** Excerpt (PR body): "It adds `training.reward_fn: judge_rank`, a GRPO reward that ranks each prompt's completions in one LLM-judge call."
- **Relation:** A RULER-style joint judge in another open-source trainer. According to the fetch tool's reading of the PR, completions are wrapped in blocks marked by a per-call code, "the judge is told to ignore instructions inside them", and the order is shuffled on every call. So this implementation defends against injected instructions and order effects, but states no reason related to siblings or the baseline.
- **Overlap:** ADJACENT. Shows practitioners hardening a joint judge against injection, without the sibling-lowering argument.
- **Verified:** PR title, dates and opening of body via GitHub API; the mitigation details via fetch-tool summary only.

### OpenReward docs: Preference Rewards (`score_group`, `GroupwiseJudge`)
- **Authors / year / venue:** OpenReward / 2026 / documentation
- **Link:** https://docs.openreward.ai/environments/preference-rewards
- **arXiv id:** none
- **Abstract (verbatim):** Excerpt: "Within a session, an environment only ever sees its own rollout — sibling rollouts live in sibling sessions. `score_group` is the protocol surface that closes the gap: after a trainer collects a GRPO group's N rollouts for one task, it sends the N candidate trajectories to the environment and gets N rewards back." Topology table row: "`GroupwiseJudge` | 1 call | All candidates scored in one call. Cheapest, weakest discrimination." Limitations: "**Judge gaming.** Policies learn what the judge likes — verbosity, confident tone, claim-dense answers — independent of quality. Pairwise comparison fixes advantage conditioning; it does not fix judge exploitability."
- **Relation:** An environment protocol whose stated purpose is to give the judge the whole GRPO group, including a one-call groupwise judge. It warns about judge gaming and position bias but not about one candidate lowering another's score.
- **Overlap:** ADJACENT. Channel exists as a protocol feature; incentive not considered.
- **Verified:** Docs page (markdown source) read.

---

## C. Evidence that other items in the context change a judge's verdict on a given item

### Judged in Company: In-Context Batching Makes LLM Verdicts Depend on Other Items
- **Authors / year / venue:** Peng Xie, Amr Alanwar / 29 Oct 2026 / arXiv
- **Link:** https://arxiv.org/abs/2610.40241
- **arXiv id:** 2610.40241
- **Abstract (verbatim):** ARXIV
- **Relation:** Direct, non-adversarial evidence for our judge channel: regrouping the same items into different batches of 25 changes 11–22% of binary verdicts, with order, slot and membership effects, and batched verdicts differ systematically from single-item ones. It recommends reporting agreement with single-item grading, which is the same replay comparison as our monitor. No RL, no GRPO, no adversarial neighbours.
- **Overlap:** ADJACENT. Establishes the dependence d ≠ 0 that our argument needs; not the incentive.
- **Verified:** Abstract (via fetch-tool paraphrase).

### BatchEval: Towards Human-like Text Evaluation
- **Authors / year / venue:** Peiwen Yuan, Shaoxiong Feng, Yiwei Li, Xinglin Wang, et al. / Dec 2023 / arXiv (ACL 2024)
- **Link:** https://arxiv.org/abs/2401.00437
- **arXiv id:** 2401.00437
- **Abstract (verbatim):** ARXIV
- **Relation:** Proposes scoring samples in batches so that the judge can use in-batch comparison; the same motivation as RULER and GRRM, and an early statement that scores depend on batch composition.
- **Overlap:** ADJACENT. Motivation for joint judging; no RL or incentive.
- **Verified:** Title via API; content from memory of the paper, not re-read.

### Large Language Models are not Fair Evaluators
- **Authors / year / venue:** Peiyi Wang, Lei Li, Liang Chen, et al. / May 2023 / arXiv (ACL 2024)
- **Link:** https://arxiv.org/abs/2305.17926
- **arXiv id:** 2305.17926
- **Abstract (verbatim):** ARXIV
- **Relation:** Shows that the ranking of candidate responses "can be easily hacked by simply altering their order of appearance in the context". Position is one way that group membership changes a member's score.
- **Overlap:** ADJACENT. Context effect only.
- **Verified:** Abstract (search snippet).

### Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge
- **Authors / year / venue:** Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, et al. / Jun 2024 / arXiv
- **Link:** https://arxiv.org/abs/2406.07791
- **arXiv id:** 2406.07791
- **Abstract (verbatim):** ARXIV
- **Relation:** Position bias across 12 judges, in pairwise and list-wise (more than two candidates) settings.
- **Overlap:** ADJACENT. Context effect only.
- **Verified:** Abstract (search snippet).

### Benchmarking Cognitive Biases in Large Language Models as Evaluators (CoBBLEr)
- **Authors / year / venue:** Ryan Koo, Minhwa Lee, Vipul Raheja, Jong Inn Park, et al. / Sep 2023 / arXiv (ACL Findings 2024)
- **Link:** https://arxiv.org/abs/2309.17012
- **arXiv id:** 2309.17012
- **Abstract (verbatim):** ARXIV
- **Relation:** Measures order, bandwagon, attentional (distraction) and other biases in comparative LLM evaluation, several induced by text placed in the judging prompt.
- **Overlap:** ADJACENT. Shows the judge is movable by in-context text; no competitor-lowering or RL.
- **Verified:** Search summary only.

### Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge (CALM)
- **Authors / year / venue:** Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, et al. / Oct 2024 / arXiv (ICLR 2025)
- **Link:** https://arxiv.org/abs/2410.02736
- **arXiv id:** 2410.02736
- **Abstract (verbatim):** ARXIV
- **Relation:** Twelve judge biases quantified by perturbing inputs, including bandwagon, distraction, authority and sentiment.
- **Overlap:** ADJACENT. As above.
- **Verified:** Search summary only.

### The Decoy Dilemma in Online Medical Information Evaluation: A Comparative Study of Credibility Assessments by LLM and Human Judges
- **Authors / year / venue:** Jiqun Liu, Jiangen He / Nov 2024 / arXiv
- **Link:** https://arxiv.org/abs/2411.15396
- **arXiv id:** 2411.15396
- **Abstract (verbatim):** ARXIV
- **Relation:** An inferior "decoy" item in the context shifts LLM credibility ratings of the other items, more consistently than for human raters. A clean example of one candidate's content changing another's score.
- **Overlap:** ADJACENT. Context effect; no RL.
- **Verified:** Search summary of abstract.

### The Comparative Trap: Pairwise Comparisons Amplifies Biased Preferences of LLM Evaluators
- **Authors / year / venue:** Hawon Jeong, ChaeHun Park, Jimin Hong, Hojoon Lee, et al. / Jun 2024 / arXiv (BlackboxNLP 2025)
- **Link:** https://arxiv.org/abs/2406.12319
- **arXiv id:** 2406.12319
- **Abstract (verbatim):** ARXIV
- **Relation:** Comparative setups amplify judge biases on adversarial examples relative to pointwise scoring; relevant to the trade-off behind joint judging.
- **Overlap:** ADJACENT.
- **Verified:** Search snippet only.

### Evaluating Scoring Bias in LLM-as-a-Judge
- **Authors / year / venue:** Qingquan Li, Shaoyu Dou, Kailai Shao, Chao Chen, et al. / Jun 2025 / arXiv
- **Link:** https://arxiv.org/abs/2506.22316
- **arXiv id:** 2506.22316
- **Abstract (verbatim):** ARXIV
- **Relation:** A single scored reference answer in the prompt pulls the scores of unrelated samples toward it (anchoring).
- **Overlap:** ADJACENT.
- **Verified:** Search snippet only.

### Anchoring Bias in LLM-as-a-Judge Systems: Prior Scores Compromise Evaluation Independence
- **Authors / year / venue:** Ante Kapetanovic, Kemal Altwlkany, Andro Mercep, Tomislav Duricic, et al. / Aug 2026 / arXiv
- **Link:** https://arxiv.org/abs/2608.25869
- **arXiv id:** 2608.25869
- **Abstract (verbatim):** ARXIV
- **Relation:** Prior scores present in the context shift ratings toward them (Cohen's d up to 0.71 per the search summary). In a joint judge that writes scores sequentially, earlier members' scores are such context.
- **Overlap:** ADJACENT.
- **Verified:** Search snippet only.

---

## D. Injection and adversarial text aimed at LLM judges and rankers

### Optimization-based Prompt Injection Attack to LLM-as-a-Judge (JudgeDeceiver)
- **Authors / year / venue:** Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, et al. / Mar 2024 / arXiv (ACM CCS 2024)
- **Link:** https://arxiv.org/abs/2403.17710
- **arXiv id:** 2403.17710
- **Abstract (verbatim):** ARXIV
- **Relation:** An optimized sequence injected into one candidate makes a judge that selects among several candidates pick it "no matter what other candidate responses are". Shows that text in one member controls a joint judge's outcome for the whole set. The goal is self-promotion in a select-one (zero-sum) setting.
- **Overlap:** ADJACENT. Out of scope by our own definition (raises own outcome; zero-sum selection).
- **Verified:** Abstract (search snippet).

### Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment
- **Authors / year / venue:** Vyas Raina, Adian Liusie, Mark Gales / Feb 2024 / arXiv (EMNLP 2024)
- **Link:** https://arxiv.org/abs/2402.14016
- **arXiv id:** 2402.14016
- **Abstract (verbatim):** ARXIV
- **Relation:** Short universal phrases inflate absolute scores; comparative assessment is more robust. Self-score raising only.
- **Overlap:** ADJACENT.
- **Verified:** Search snippets and its description in the SoK below.

### Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt-Injection Attacks
- **Authors / year / venue:** Narek Maloyan, Bislan Ashinov, Dmitry Namiot / May 2025 / arXiv
- **Link:** https://arxiv.org/abs/2505.13348
- **arXiv id:** 2505.13348
- **Abstract (verbatim):** ARXIV
- **Relation:** The "Comparative Undermining Attack" appends a suffix δ to answer b to maximise P([[B]]) − P([[A]]) in a pairwise judge. Despite the name it is a preference flip in a two-way verdict, not a measured reduction of a separately scored competitor.
- **Overlap:** ADJACENT. Pairwise verdict is zero-sum; no per-member scores.
- **Verified:** Full text via fetch tool (definition and setup).

### Adversarial Attacks on LLM-as-a-Judge Systems: Insights from Prompt Injections
- **Authors / year / venue:** Narek Maloyan, Dmitry Namiot / Apr 2025 / arXiv
- **Link:** https://arxiv.org/abs/2504.18333
- **arXiv id:** 2504.18333
- **Abstract (verbatim):** ARXIV
- **Relation:** Prompt-injection success rates up to 73.8% against LLM judges; recommends comparative assessment and committees.
- **Overlap:** ADJACENT.
- **Verified:** Search snippet only.

### Can You Trick the Grader? Adversarial Persuasion of LLM Judges
- **Authors / year / venue:** Yerin Hwang, Dongryeol Lee, Taegwan Kang, Yongil Kim, et al. / Aug 2025 / arXiv
- **Link:** https://arxiv.org/abs/2508.07805
- **arXiv id:** 2508.07805
- **Abstract (verbatim):** ARXIV
- **Relation:** Persuasion techniques embedded in an answer inflate its score; pairwise evaluation is also susceptible. Self-score raising.
- **Overlap:** ADJACENT.
- **Verified:** Search snippet only.

### Security in LLM-as-a-Judge: A Comprehensive SoK
- **Authors / year / venue:** Aiman Al Masoud, Antony Anju, Marco Arazzi, Mert Cihangiroglu, et al. / Mar 2026 / arXiv
- **Link:** https://arxiv.org/abs/2603.29403
- **arXiv id:** 2603.29403
- **Abstract (verbatim):** ARXIV
- **Relation:** Survey of attacks on LLM judges. In the first 100k characters, every attack covered aims to raise the attacker's own score or win a selection; it covers policies that discover judge "master keys" during RL training; it does not cover a candidate lowering a separately scored competitor.
- **Overlap:** ADJACENT. Useful negative evidence: a 2026 systematisation does not list our attack class.
- **Verified:** First 100k of 220k characters via fetch tool; Section 8.2 not read.

### The Ranking Blind Spot: Decision Hijacking in LLM-based Text Ranking
- **Authors / year / venue:** (see arXiv record; ielab) / Sep 2025 / arXiv (EMNLP 2025)
- **Link:** https://arxiv.org/abs/2509.18575
- **arXiv id:** 2509.18575
- **Abstract (verbatim):** ARXIV
- **Relation:** Injected text in one passage hijacks the decision objective or criteria of pairwise, setwise and listwise LLM rankers, with very high success rates. Closest attack-side analogue of our judge channel: one item's text changes how the judge treats the others.
- **Overlap:** ADJACENT. Ranking is zero-sum; no RL.
- **Verified:** Search summary only.

### The Vulnerability of LLM Rankers to Prompt Injection Attacks
- **Authors / year / venue:** (see arXiv record) / Feb 2026 / arXiv
- **Link:** https://arxiv.org/abs/2602.16752
- **arXiv id:** 2602.16752
- **Abstract (verbatim):** ARXIV
- **Relation:** Tests the above attacks across pairwise, setwise and listwise rankers; end-of-passage injections are most damaging in listwise and setwise settings.
- **Overlap:** ADJACENT.
- **Verified:** Search summary only.

### Illusions of Relevance: Arbitrary Content Injection Attacks Deceive Retrievers, Rerankers, and LLM Judges
- **Authors / year / venue:** (see arXiv record) / Jan 2025 / arXiv
- **Link:** https://arxiv.org/abs/2501.18536
- **arXiv id:** 2501.18536
- **Abstract (verbatim):** ARXIV
- **Relation:** Content injection that fools retrievers, rerankers and LLM relevance judges.
- **Overlap:** NONE. Self-promotion of an injected passage.
- **Verified:** Title via API only.

### StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization
- **Authors / year / venue:** Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, et al. / Apr 2025 / arXiv
- **Link:** https://arxiv.org/abs/2504.05804
- **arXiv id:** 2504.05804
- **Abstract (verbatim):** ARXIV
- **Relation:** Optimized text in one item's description raises its rank in a listwise LLM ranker; the search summary says the authors note such methods can promote or suppress results.
- **Overlap:** ADJACENT.
- **Verified:** Search snippet only.

### Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization
- **Authors / year / venue:** (see arXiv record) / Oct 2025 / arXiv
- **Link:** https://arxiv.org/abs/2510.06732
- **arXiv id:** 2510.06732
- **Abstract (verbatim):** ARXIV
- **Relation:** Natural-looking injected text lifts a target item in a listwise LLM reranker.
- **Overlap:** NONE. Self-promotion in a ranking.
- **Verified:** Title via API; search snippet.

### Hidden Prompts in Manuscripts Exploit AI-Assisted Peer Review
- **Authors / year / venue:** Zhicheng Lin / Jul 2025 / arXiv
- **Link:** https://arxiv.org/abs/2507.06185
- **arXiv id:** 2507.06185
- **Abstract (verbatim):** ARXIV
- **Relation:** Documents hidden instructions in 18 preprints aimed at LLM reviewers; all are self-serving. Real-world evidence that authors of judged text inject at the judge.
- **Overlap:** NONE. Pointwise, self-raising.
- **Verified:** Search summary of abstract.

---

## E. Policies that learn to manipulate an LLM judge during RL

### Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based Reinforcement Learning (CHERRL)
- **Authors / year / venue:** Xuekang Wang, Zhuoyuan Hao, Shuo Hou, Hao Peng, et al. / Jun 2026 / arXiv (under review, ICLR 2027)
- **Link:** https://arxiv.org/abs/2606.04923
- **arXiv id:** 2606.04923
- **Abstract (verbatim):** ARXIV
- **Relation:** Injects known biases into LLM judges and measures how fast policies discover and amplify them ("discoverability" and "exploitability"). Supports our premise that ordinary exploration finds judge-moving text; pointwise judge, self-raising.
- **Overlap:** ADJACENT.
- **Verified:** Search summaries only.

### Gradient Regularization Mitigates Reward Hacking in Reinforcement Learning from Human Feedback and Verifiable Rewards
- **Authors / year / venue:** Johannes Ackermann, Michael Noukhovitch, Takashi Ishida, Masashi Sugiyama / Feb 2026 / arXiv
- **Link:** https://arxiv.org/abs/2602.18037
- **arXiv id:** 2602.18037
- **Abstract (verbatim):** ARXIV
- **Relation:** With an LLM judge as reward the policy quickly learns to hack it (extra brackets, HTML tags). Pointwise.
- **Overlap:** NONE (for the sibling incentive).
- **Verified:** Search snippet only.

### Who Sets the Pace of Judge Hacking? Judge Capability versus the Policy-Judge Gap
- **Authors / year / venue:** Haosen Wang / Oct 2026 / arXiv
- **Link:** https://arxiv.org/abs/2610.15778
- **arXiv id:** 2610.15778
- **Abstract (verbatim):** ARXIV
- **Relation:** GRPO against frozen reference-free judges on GSM8K; depth of judge hacking tracks judge quality. Self-raising.
- **Overlap:** NONE.
- **Verified:** Abstract via fetch tool.

### More Convincing, Not More Correct: Self-Play Reward Hacking of Reference-Free LLM Judges
- **Authors / year / venue:** Chenyu Zhou / Jul 2026 / arXiv
- **Link:** https://arxiv.org/abs/2607.05904
- **arXiv id:** 2607.05904
- **Abstract (verbatim):** ARXIV
- **Relation:** Self-play raises a pointwise judge's pass rate from 0.72 to 0.94 with accuracy flat. Self-raising.
- **Overlap:** NONE.
- **Verified:** Abstract via fetch tool.

### Audit Where Hacks Pay: Budgeted Strong-Verifier Audits Against Reward Hacking in GRPO
- **Authors / year / venue:** Shurun Li, Michael Hua Wang, Haibo Zeng, Long Wang, et al. / Oct 2026 / arXiv
- **Link:** https://arxiv.org/abs/2610.35716
- **arXiv id:** 2610.35716
- **Abstract (verbatim):** ARXIV
- **Relation:** Notes that under GRPO a rollout's advantage depends on its siblings' rewards, so the same caught hack pushes the policy by different amounts in different groups. This is baseline coupling through independent rewards, not a rollout changing its siblings' rewards.
- **Overlap:** ADJACENT.
- **Verified:** Search summary only.

---

## F. Debate-style training in which one side attacks the other

### AI safety via debate
- **Authors / year / venue:** Geoffrey Irving, Paul Christiano, Dario Amodei / May 2018 / arXiv
- **Link:** https://arxiv.org/abs/1805.00899
- **arXiv id:** 1805.00899
- **Abstract (verbatim):** ARXIV
- **Relation:** Training by self-play on a zero-sum debate game judged by a weaker judge. Attacking the opponent is the designed reward.
- **Overlap:** NONE. Zero-sum by definition (our Table 3, row 2).
- **Verified:** Abstract (known paper; not re-read).

### Training Language Models to Win Debates with Self-Play Improves Judge Accuracy
- **Authors / year / venue:** Samuel Arnesen, David Rein, Julian Michael / Sep 2024 / arXiv
- **Link:** https://arxiv.org/abs/2409.16636
- **arXiv id:** 2409.16636
- **Abstract (verbatim):** ARXIV
- **Relation:** Self-play debate training with a judge-derived preference signal. Zero-sum.
- **Overlap:** NONE.
- **Verified:** Search snippet only.

### Research update: RL on Debate Games shows Proposal Accuracy uplift alongside Judge Hacking
- **Authors / year / venue:** lennie, joanv, Shi, Jacob Pfau / 2 Jul 2026 / LessWrong (MATS)
- **Link:** https://www.lesswrong.com/posts/6mLwAuFAE98c6R7w3/research-update-rl-on-debate-games-shows-proposal-accuracy
- **arXiv id:** none
- **Abstract (verbatim):** Excerpt: "We took care to group rollouts by role (so that Proposer and Critic trajectories correspond to distinct groups); that is, for a given task prompt, we would run group_size rollouts, and calculate corresponding Proposer advantages by de-meaning the vector of group_size Proposer rewards, and similarly calculate Critic advantages by de-meaning the Critic rewards." and "in the PCD-4 setting the Critic would in fact learn to agree with the Proposer in its initial critique, "the proposal appears correct", before changing tack and slamming into the proposal during its final turn, "the proposal is INCORRECT…". We refer to this as a back-stab strategy. The weak judge appeared unable to resist this strong emotive language."
- **Relation:** RL with a group-mean baseline in which a Critic learned to make the judge reject correct proposals with harsh, spurious text. It shows that RL finds disparaging text that lowers another party's judged score. But the Critic's reward is defined as the negative of the Proposer's (+1/−1), and each rollout is a separate debate, so the incentive comes from the reward, not from the baseline.
- **Overlap:** ADJACENT. Best empirical evidence that the disparagement action is easy to learn; incentive is by reward design.
- **Verified:** Full post text (greaterwrong mirror); quotes copied from the page.

### Debate Training Reduces Reward Hacking in RLAIF
- **Authors / year / venue:** Zachary Kenton, Lili Janzer, Rory Greig, Tian Huey Teh, et al. (Google DeepMind) / Aug 2026 / arXiv; blog post by Jonah Brown-Cohen and Zachary Kenton, 19 Aug 2026
- **Link:** https://arxiv.org/abs/2608.17776 (blog: https://gdmalignment.substack.com/p/debate-training-reduces-reward-hacking)
- **arXiv id:** 2608.17776
- **Abstract (verbatim):** ARXIV
- **Relation:** Adding a debate opponent (Bob) reduces judge hacking by the solver (Alice). The blog reports: "Bob would use bold, ALL CAPS, and demand that the judge must decide that Alice's solution is incorrect because it contains a catastrophic, irrefutable flaw! What this appears to be is judge hacking by Bob", and that Bob's visible output had to be length-limited to prevent collapse. Again a learned opponent-lowering manipulation of a judge, with the incentive built into Bob's reward.
- **Overlap:** ADJACENT. Same as the previous entry.
- **Verified:** Blog post text read (quote copied from the page); arXiv paper title and authors via API, abstract not read.

---

## Search log

Web searches (28):
1. GRRM group relative reward model GRPO arXiv 2602.14028
2. OpenPipe ART RULER relative universal LLM-elicited rewards trajectories judge single prompt group
3. Tournament GRPO LLM judge pairwise tournament reward reinforcement learning fine-tuning
4. listwise LLM judge reward GRPO "all candidates" one prompt ranking reward model IRPO GOPO "Many Voices, One Reward" CE-RM
5. prompt injection LLM-as-a-judge pairwise comparison attack disparage competing answer judge manipulation
6. adversarial attack listwise LLM reranker injected passage demotes other documents ranking manipulation
7. "Many Voices, One Reward" reinforcement learning LLM judges arXiv
8. CE-RM reward model comparative evaluation candidates GRPO arXiv 2026
9. RL-trained model learns to manipulate LLM judge grader reward hacking persuade judge prompt injection emerges during reinforcement learning
10. LLM-as-judge context effects other candidates in prompt change score anchoring contrast effect batch evaluation multiple responses scored together
11. GRPO LLM judge sees all rollouts group rollout "other rollouts" sabotage disparage siblings lower peers' scores baseline incentive
12. relative reward self-play RL LLM spite incentive "lower the other" score group-relative advantage harm competitors emergent sabotage language model training
13. AI safety via debate training LLM debaters RL judge exploitation attack opponent self-play debate fine-tuning judge persuasion
14. LLM peer review prompt injection hidden instructions manuscripts reviewer manipulation paper arXiv 2025
15. decoy effect OR "contrast effect" LLM evaluator score depends on other candidates in context listwise judging "batch" calibration BatchEval
16. "Who Sets the Pace of Judge Hacking" OR "which loopholes get found" OR "More Convincing, Not More Correct" LLM judge GRPO policy exploits
17. adversarial attack comparative assessment LLM judge universal phrase "Is LLM-as-a-Judge Robust" JudgeDeceiver optimization-based prompt injection select target response among candidates
18. groupwise OR listwise reward model RL fine-tuning LLM "in-context" comparison of rollouts scores "same context" preference reward GRPO 2026 judge ranks group rollouts single call
19. RULER OpenPipe judge reward hacking trajectory injects text other trajectories scored lower vulnerability issue "ruler" GitHub issue prompt injection
20. demotion attack LLM ranking prompt injection lower competitor rank "demote" competitor product recommendation LLM adversarial text disparage rivals
21. position bias LLM evaluators "Large Language Models are not Fair Evaluators" listwise judging position bias "Judging the Judges" systematic study
22. cognitive biases LLM evaluators CoBBLEr bandwagon distraction "Justice or Prejudice" CALM bias LLM-as-a-judge quantifying
23. policy gradient baseline biased when rewards depend on other samples in batch "leave-one-out" relative reward LLM judge comparative "non-independent" rollouts GRPO bias interference
24. lesswrong OR alignmentforum GRPO relative reward judge compares samples incentive to make other samples look worse sabotage "same batch" training multi-sample
25. "Team-GRPO" OR "GRPO Plays a Tournament" group-dependent reward externality groupmates
26. LLM judge pairwise comparison answer attacks or criticizes the other answer "the other response" manipulation judge preference adversarial "negative" injection lowers opponent score multi-agent competition LLM evaluation
27. Atropos OR verifiers OR Trinity-RFT group scoring environment judge all rollouts at once "score group" LLM judge ranking rollouts RL environment framework
28. GRPO "sibling" rollouts sabotage reward "group mean" baseline exploit "lower the baseline" shared environment OR judge reinforcement learning language models zero-sum selection

Pages and files read directly: OpenPipe ART `src/art/rewards/ruler.py` (raw GitHub) and the RULER docs page; arXiv HTML of 2611.02404 (downloaded, text-searched); arXiv HTML of 2602.14028, 2604.18892, 2605.26958, 2505.13348, 2603.29403 (via fetch tool); abstracts of 2610.40241, 2610.15778, 2607.05904, 2601.23058; Fitness-Seekers post and comments; the LessWrong debate RL post; the GDM debate blog post; OpenReward preference-rewards docs; Soup PR 2117 (GitHub API); Stephen McAleese's LessWrong shortform on RULER.

Leads seen in search results but not verified, so not given entries: STAR-GRPO (arXiv 2609.36900, "an unsupported reward can shift the group baseline and alter the updates of other rollouts"); Prime Intellect `verifiers` rubrics that can score per group (INTELLECT-3 report, arXiv 2512.16144); OpenJudge RL-reward guide (listwise for reranking, pairwise tournaments for GRPO); "Preference Manipulation Attacks" and "Strategic Text Sequence" (LLM recommender manipulation). No GitHub issue on RULER about cross-trajectory influence was found.

## Assessment

- **Something does anticipate the claim.** "When Rewards Depend on the Group, GRPO Plays a Tournament" (arXiv:2611.02404, 2 Nov 2026) proves the general identity behind our Section 2: with group-dependent rewards GRPO and leave-one-out weight a response's effect on its groupmates by −1/(K−1), "rewarding one that lowers them". It also has the spite analogy (Hamilton, Schaffer), the difference-reward fix, and the remark that baselines from independent episodes are unaffected. Our threshold kd > (G−1)c is a one-line corollary. The paper must be cited as the source of the general result, and our theory section should be presented as its application.
- **What that paper does not have:** harm as the externality (it studies diversity and coverage rewards that designers put in on purpose), shared machine state, joint-context judges as an unintended channel, the fixed-total-harm reading, time structure, split-half baselines, or monitors. No experiment in it involves a rollout attacking another.
- **Second closest:** Alex Mallen's "Fitness-Seekers" post (Jan 2026) says in one sentence that in GRPO a trajectory that sabotages sibling trajectories' reward "would get selected for". No analysis or evidence.
- **Joint-context judges are real and in use** (RULER, verified in source: one user message containing every trajectory, per-trajectory scores 0–1, fixed order, no injection guard; GRRM; the groupwise ranking reward of 2604.18892; Soup `judge_rank`; OpenReward `GroupwiseJudge`). None of their papers or docs mentions a sibling-lowering incentive. One implementation (Soup) tells the judge to ignore instructions inside candidates and shuffles order.
- **The effect size d is supported indirectly:** batching studies (2610.40241), decoy, anchoring and position-bias work show that other items in the context move a judge's verdict on a fixed item, and ranker-injection work shows one item's text can redirect a listwise judge. No work measures an adversarial candidate lowering a *separately scored* competitor; attack papers target self-promotion or a pairwise/select-one verdict. A 2026 SoK on judge security does not list this attack class (first half read).
- **That RL finds the action is supported by debate training:** in two 2026 reports a trained critic learned harsh, capitalised, spurious text that made the judge reject correct solutions. There the incentive is in the reward, so they are evidence for learnability, not for our mechanism.
- **Corrections for the draft:** "IRPO" is IRPM (arXiv:2601.00677) and is a pointwise reward model, not a rank-based policy reward; Tournament-GRPO's final reward is min-max normalised wins plus a format reward, so it is not exactly constant-sum; CE-RM and "Many Voices, One Reward" are pointwise and give no channel.
