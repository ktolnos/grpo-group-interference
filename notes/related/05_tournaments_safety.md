# Related work 05: tournaments / relative performance evaluation, and AI-safety work on training incentives

Searched 11 November 2026. Part A = economics of relative performance evaluation and sabotage.
Part B = AI safety. Entries are ordered by relevance within each part.

**Headline:** one AI-safety post states our qualitative claim in one sentence (Mallen, "Fitness-Seekers",
Jan 2026, appendix). Nothing found gives the condition, the group-size dependence, the fixes or an
experiment. In economics, Konrad (2000) is the closest formal analogue of the group-size result.

---

# Part A. Economics: relative performance evaluation and sabotage

### Sabotage in Rent-Seeking Contests
- **Authors / year / venue:** Kai A. Konrad, 2000, Journal of Law, Economics, and Organization 16(1), 155-165.
- **Link:** https://doi.org/10.1093/jleo/16.1.155 (abstract read at https://ideas.repec.org/a/oup/jleorg/v16y2000i1p155-65.html)
- **arXiv id:** none
- **Abstract (verbatim):** This article considers the interaction between two types of effort in lobbying contests: effort that improves the contestant's own performance (standard rent seeking), and effort that reduces particular rivals' performance (sabotage). Due to a positive externality, sabotage is a "small number" phenomenon. Sabotage may increase lobbying efforts and the dissipation rate in lobbying contests compared to a situation in which sabotage is not feasible.
- **Relation:** The closest formal analogue of our group-size result. Sabotage aimed at one rival helps every other contestant, so the saboteur captures only a share of the benefit that shrinks with the number of contestants; with enough contestants nobody sabotages. This is our "total harm fixed" case, D > (G-1)c: interference pays only in small groups. Differences: his contest is zero-sum by construction (a prize), ours is an artefact of the estimator; his agents optimise, ours are updated by a gradient.
- **Overlap:** ADJACENT - same small-number structure, in a human contest with a real prize, not a learning algorithm's baseline.
- **Verified:** abstract (RePEc). The mechanism and the "no sabotage if the number of contestants is sufficiently high" statement were read second-hand in Amegashie's survey (next entry), full text. The paper's own formal condition was NOT read; read it before quoting a formula.

### Sabotage in Contests: An Overview
- **Authors / year / venue:** J. Atsu Amegashie, 2013, CESifo Working Paper No. 4422.
- **Link:** https://www.ifo.de/DocDL/cesifo1_wp4422.pdf
- **arXiv id:** none
- **Abstract (verbatim):** This paper reviews the relatively small literature on sabotage in contests. It looks at both the formal game-theoretic literature and the empirical and experimental literatures. The treatment is intended to be intuitive with minimal use of technical jargon.
- **Relation:** Gives the clearest statement of why sabotage falls with the number of contestants. Excerpt (p. 4, verbatim): "unlike the standard rent-seeking effort, sabotage has a public-good characteristic. When a player in a contest increases his rent-seeking effort, it increases his probability of winning and all the benefits accrue to him alone. In contrast, when a player weakens a specific player by sabotaging him, all other players (including the saboteur) benefit. Konrad (2000) refers to this as the dispersion effect of sabotage. Because of this effect, the benefit per player is smaller the higher is the number of players in the game. Accordingly, players have a greater incentive to invest more effort in sabotage if the total number of players is small." Footnote 8 (verbatim): "The model in Amegashie (2012) also yields the result, first discovered by Konrad (2000), that there is no investment in sabotage if the number of contestants is sufficiently high."
- **Overlap:** ADJACENT - states the dispersion effect (our 1/G factor) for contests.
- **Verified:** full text (PDF converted to text).

### Sabotage in contests: a survey
- **Authors / year / venue:** Subhasish M. Chowdhury, Oliver Gürtler, 2015, Public Choice 164(1), 135-155.
- **Link:** https://doi.org/10.1007/s11127-015-0264-9
- **arXiv id:** none
- **Abstract (verbatim):** A contest is a situation in which individuals expend irretrievable resources to win valuable prize(s). “Sabotage” is a deliberate and costly act of damaging a rival’s likelihood of winning the contest. Sabotage can be observed in, e.g., sports, war, promotion tournaments, and political or marketing campaigns. In this article, we provide a model of, and different perspectives on, such sabotage activities and review the economics literature analyzing the act of sabotage in contests. We discuss the theories and evidence emphasizing the means of sabotage, why sabotage occurs, and the effects of sabotage on individual contestants and on overall welfare, with possible means of reducing sabotage. We note that most sabotage activities are aimed at the ablest contestant, the possibility of sabotage reduces productive effort exerted by the contestants, and sabotage may lessen the effectiveness of public policies, such as affirmative action, or information revelation in contests. We discuss the different policies a designer might use to counteract sabotage. We conclude by pointing out some areas of future research.
- **Relation:** The standard survey to cite for "relative evaluation induces sabotage". Its unified model is where to look for a first-order condition to set beside ours.
- **Overlap:** ADJACENT - survey of the human-contest literature.
- **Verified:** abstract only (Semantic Scholar API). The model section was not read.

### Pay Equality and Industrial Politics
- **Authors / year / venue:** Edward P. Lazear, 1989, Journal of Political Economy 97(3), 561-580.
- **Link:** https://doi.org/10.1086/261616
- **arXiv id:** none
- **Abstract (verbatim):** Personnel managers often argue that equitable pay treatment manifested as wage compression is useful because it reduces disharmony among workers. But it is far from obvious that a compressed salary structure is morale improving since better workers may feel disenchanted by this scheme. However, when workers' rewards are based on relative comparisons, salary compression reduces uncooperative behavior that is detrimental to the firm. Relative comparisons imply that some reference group must be selected. The major result is that, within the relevant groups, some wage compression is efficient.
- **Relation:** The founding statement that pay based on relative comparison rewards harming co-workers. His remedy, shrinking the prize spread, corresponds to shrinking the weight on the relative term; ours removes the dependence of the baseline on the actor. His remark that "some reference group must be selected" is the same design choice as our choice of baseline group.
- **Overlap:** ADJACENT - same incentive, in labour contracts, two-worker model, no group-size condition.
- **Verified:** abstract only (RePEc).

### How Many Winners are Good to Have? On Tournaments with Sabotage
- **Authors / year / venue:** Christine Harbring, Bernd Irlenbusch, 2008, Journal of Economic Behavior & Organization (earlier as IZA DP 1777); volume and pages not checked.
- **Link:** https://doi.org/10.1016/j.jebo.2006.03.004
- **arXiv id:** none
- **Abstract (verbatim):** From an employer's perspective, a tournament should induce agents to exert productive activities but refrain from destructive ones. We experimentally test the predictive power of a tournament model which suggests that (within a reasonable framework) productive and destructive activities are influenced neither by the number of agents taking part in the tournament nor by the fraction of the winner prizes. Our results clearly confirm that sabotage in tournaments indeed occurs. While tournament size has virtually no effect on behavior, a balanced fraction of winner and loser prizes appears to enhance productive activities.
- **Relation:** The counterpart of our "harm per target fixed" case. Here sabotage lowers the output of all other contestants, and the model predicts no dependence on tournament size, as our d > c when k = G-1. Together with Konrad (2000) this gives both of our Section 2.4 cases in the economics literature: size-independent when everyone is hit, small-number when the harm is targeted.
- **Overlap:** ADJACENT - group-size prediction (and a lab test) for human tournaments.
- **Verified:** abstract only (Semantic Scholar API). That sabotage hits all rivals is from a search-result summary of the paper; check the model before citing it.

### Sabotage in Tournaments: Evidence from a Laboratory Experiment
- **Authors / year / venue:** Christine Harbring, Bernd Irlenbusch, 2011, Management Science 57(4), 611-627.
- **Link:** https://doi.org/10.1287/mnsc.1100.1296
- **arXiv id:** none
- **Abstract (verbatim):** Although relative performance schemes are pervasive in organizations, reliable empirical data on induced sabotage behavior are almost nonexistent. We study sabotage in repeated tournaments in a controlled laboratory experiment and observe that effort and sabotage are higher for higher wage spreads. Additionally, we find that also in the presence of tournament incentives, agents react reciprocally to higher wages by exerting higher effort. Destructive activities are reduced by explicitly calling them by their name “sabotage.” Communication among principal and agents can curb sabotage when they agree on flat prize structures and increased output. If sabotage is not possible, the principals choose tournament incentives more often.
- **Relation:** Laboratory evidence that relative pay induces sabotage and that it grows with the prize spread.
- **Overlap:** ADJACENT - empirical, human subjects.
- **Verified:** abstract only (Semantic Scholar API).

### Performance Appraisals and the Impact of Forced Distribution - An Experimental Investigation
- **Authors / year / venue:** Johannes Berger, Christine Harbring, Dirk Sliwka, 2013, Management Science 59(1), 54-68.
- **Link:** https://doi.org/10.1287/mnsc.1120.1624
- **arXiv id:** none
- **Abstract (verbatim):** A real-effort experiment is investigated in which supervisors have to rate the performance of individual workers who in turn receive a bonus payment based on these ratings. We compare a baseline treatment in which supervisors are not restricted in their rating behavior to a forced distribution system in which they have to assign differentiated grades. We find that productivity is significantly higher under a forced distribution by about 6% to 12%. However, the productivity effects are less clear cut when participants have prior experience with the baseline condition. Moreover, a forced distribution becomes detrimental when workers have access to a simple option to sabotage each other.
- **Relation:** The forced-ranking ("stack ranking") result: grading against the group helps until members can harm each other, then it lowers output. Same sign reversal as our mean-reward collapse.
- **Overlap:** ADJACENT - the best single citation for forced ranking / grading on a curve with sabotage.
- **Verified:** abstract only (Semantic Scholar API).

### Tournaments and Office Politics: Evidence from a Real Effort Experiment
- **Authors / year / venue:** Jeffrey P. Carpenter, Peter Matthews, and a third author listed by Semantic Scholar as "Johannes Schirm" (I believe the journal has John Schirm - check), 2010, American Economic Review 100(1), from p. 504.
- **Link:** https://doi.org/10.1257/aer.100.1.504
- **arXiv id:** none
- **Abstract (verbatim):** In many environments, tournaments can elicit more effort from workers, except perhaps when workers can sabotage each other. Because it is hard to separate effort, ability and output in many real workplace settings, the empirical evidence on the incentive effect of tournaments is thin. There is even less evidence on the impact of sabotage because real world acts of sabotage are often subtle manifestations of subjective peer evaluation or "office politics." We discuss a real effort experiment in which effort, quality adjusted output and office politics are compared under piece rates and tournaments. Our results suggest that tournaments increase effort only in the absence of office politics. Competitors are more likely to sabotage each other in tournaments and, as a result, workers actually provide less effort simply because they expect to be the victims of sabotage. Adjusting output for quality with the rating of an independent auditor shrinks the incentive effect of the tournament even further since output tends to become more slipshod.
- **Relation:** Sabotage here works through peer evaluation, close to our joint-judging channel. The "independent auditor" is an evaluation the contestants cannot influence, like our independent baseline.
- **Overlap:** ADJACENT.
- **Verified:** abstract only (Semantic Scholar API).

### Sabotage in Promotion Tournaments
- **Authors / year / venue:** Kong-Pin Chen, 2003, Journal of Law, Economics, and Organization 19(1), 119-140.
- **Link:** https://ideas.repec.org/a/oup/jleorg/v19y2003i1p119-140.html
- **arXiv id:** none
- **Abstract (verbatim):** This article studies the nature, determinants, and impact of "negative" activities in organizations. In competing for promotion, the members in organizations can work not only to enhance their own performances, but also to "sabotage" their opponent's performances. It is worthwhile for them to engage in negative activities because promotion is generally based on relative, rather than absolute, performance, and its nature is winner take all. I find that abler members are subject to more attacks. Consequently, not only is there a double inefficiency in effort, but also members of the highest caliber might not have the best chance of being promoted. Finally, I discuss several institutional designs that might help to reduce the influence of negative activities.
- **Relation:** States the reason in our terms: sabotage pays "because promotion is generally based on relative, rather than absolute, performance". Adds who gets targeted (the ablest), which we do not model.
- **Overlap:** ADJACENT.
- **Verified:** abstract only (RePEc).

### Selection Tournaments, Sabotage, and Participation
- **Authors / year / venue:** Johannes Münster, 2007, Journal of Economics & Management Strategy 16(4), 943-970.
- **Link:** https://doi.org/10.1111/j.1530-9134.2007.00163.x
- **arXiv id:** none
- **Abstract (verbatim):** This paper studies sabotage in tournaments with at least three contestants, where the contestants know each other well. Every contestant has an incentive to direct sabotage specifically against his most dangerous rival. In equilibrium, contestants who choose a higher productive effort are sabotaged more heavily. This might explain findings from psychology, where victims of mobbing are sometimes found to be overachieving. Moreover, sabotage equalizes promotion chances. The effect is most pronounced if the production functions are linear in sabotage, and the cost functions depend only on the sum of all sabotage activities: in an interior equilibrium, who will win is a matter of chance, even when contestants differ a great deal in their abilities. This, in turn, has adverse consequences for who might want to participate in a tournament. Because better contestants anticipate that they will be sabotaged more strongly, it may happen that the most able stay out and the tournament selects one of the less able with probability one. I also study the case where some contestants are easy victims, that is, easier to sabotage than others.
- **Relation:** n >= 3 tournament with targeted sabotage. "Sabotage equalizes" outcomes: under GRPO, compressed within-group rewards also shrink the learning signal.
- **Overlap:** ADJACENT.
- **Verified:** abstract only (RePEc).

### Incentives for Helping on the Job: Theory and Evidence
- **Authors / year / venue:** R. Drago, G. Garvey (initials as given by Semantic Scholar), 1998, Journal of Labor Economics.
- **Link:** https://doi.org/10.1086/209880
- **arXiv id:** none
- **Abstract (verbatim):** Recent advances in incentive theory stress the multidimensional nature of agent effort and specifically cases where workers affect one anothers' performance through “helping” efforts. This article models helping efforts as determined by the compensation package and task allocation. The model is tested with Australian evidence on reported helping efforts within work groups. The evidence consistently supports the hypothesis that helping efforts are reduced, while individual efforts are increased, when promotion incentives are strong. Piece rates and profit‐sharing appear to have little effect on helping efforts, while task variety and helping efforts are positively correlated.
- **Relation:** The mirror image: a group-relative baseline also penalises helping peers (an action with H < 0 gets a negative contribution to its action gap). We do not currently say this; it follows from Table 1.
- **Overlap:** ADJACENT.
- **Verified:** abstract only (Semantic Scholar API).

### Sabotage in Tournaments: Evidence from a Natural Experiment
- **Authors / year / venue:** Loukas Balafoutas, Florian Lindner, Matthias Sutter, 2012, Kyklos 65(4), 425-441.
- **Link:** https://ideas.repec.org/a/bla/kyklos/v65y2012i4p425-441.html
- **arXiv id:** none
- **Abstract (verbatim):** Many tournaments are plagued by sabotage among competitors. Typically, sabotage is welfare-reducing, but from an individual's perspective an attractive alternative to exerting positive effort. Yet, given its illegal and often immoral nature, sabotage is typically hidden, making it difficult to assess its extent and its victims. Therefore, we use data from Judo World Championships, where a rule change in 2009 basically constituted a natural experiment that introduced one costless opportunity for sabotage. In Judo, competitors can break an opponent's attack in an unsportsmanlike manner; these are seen as acts of sabotage. Based on a unique dataset of 1,422 fights, we find that the rule change in 2009 has led to a large increase in the use of sabotage. Moreover, sabotage is more likely to be employed by relatively less qualified individuals, and to be targeted at more qualified ones. From a survey among spectators, we show that sabotage is welfare-reducing.
- **Relation:** Field evidence that costless sabotage (c = 0) is taken up at once in a pairwise contest (G = 2), as our condition predicts.
- **Overlap:** ADJACENT.
- **Verified:** abstract only (RePEc; the page notes the abstract is borrowed from the working-paper version).

### Rank-Order Tournaments as Optimum Labor Contracts
- **Authors / year / venue:** Edward P. Lazear, Sherwin Rosen, 1981, Journal of Political Economy 89(5), 841-864.
- **Link:** https://doi.org/10.1086/261010
- **arXiv id:** none
- **Abstract (verbatim):** This paper analyzes compensation schemes which pay according to an individual's ordinal rank in an organization rather than his output level. When workers are risk neutral, it is shown that wages based upon rank induce the same efficient allocation of resources as an incentive reward scheme based on individual output levels. Under some circumstances, risk-averse workers actually prefer to be paid on the basis of rank. In addition, if workers are heterogeneous inability, low-quality workers attempt to contaminate high-quality firms, resulting in adverse selection. However, if ability is known in advance, a competitive handicapping structure exists which allows all workers to compete efficiently in the same organization.
- **Relation:** The case for relative pay when workers cannot affect each other: rank pay is as efficient as output pay. This is GRPO's situation with independent rollouts. No sabotage in the model.
- **Overlap:** NONE - background; the independence assumption is implicit.
- **Verified:** abstract only (RePEc; the page notes it is borrowed from the NBER working-paper version, and "inability" is the source's typo for "in ability").

### Moral Hazard in Teams
- **Authors / year / venue:** Bengt Holmström, 1982, Bell Journal of Economics 13(2), 324-340.
- **Link:** https://doi.org/10.2307/3003457
- **arXiv id:** none
- **Abstract (verbatim):** This article studies moral hazard with many agents. The focus is on two features that are novel in a multiagent setting: free riding and competition. The free-rider problem implies a new role for the principal: administering incentive schemes that do not balance the budget. This new role is essential for controlling incentives and suggests that firms in which ownership and labor are partly separated will have an advantage over partnerships in which output is distributed among agents. A new characterization of informative (hence valuable) monitoring is derived and applied to analyze the value of relative performance evaluation. It is shown that competition among agents (due to relative evaluations) has merit solely as a device to extract information optimally. Competition per se is worthless. The role of aggregate measures in relative performance evaluation is also explored, and the implications for investment rules are discussed.
- **Relation:** The economic version of "a baseline is only a variance-reduction device": relative evaluation is valuable solely because peers' outcomes carry information about common noise. That is exactly what the group mean is for in GRPO. Also the source for "budget balance is the problem": advantages that sum to zero within the group are a balanced budget.
- **Overlap:** ADJACENT - explains what the baseline is for; no sabotage.
- **Verified:** abstract only (RePEc).

### Evolutionary equilibrium in Tullock contests: spite and overdissipation
- **Authors / year / venue:** Burkhard Hehenkamp, Wolfgang Leininger, Alex Possajennikov, 2004, European Journal of Political Economy 20(4), 1045-1057.
- **Link:** https://doi.org/10.1016/j.ejpoleco.2003.09.002
- **arXiv id:** none
- **Abstract (verbatim):** Tullock's analysis of rent-seeking as a contest is reconsidered from an evolutionary point of view. We show that evolutionary stable behavior in a Tullock contest exists and differs from behavior in Nash equilibrium. Evolutionary stable behavior in these contests is robust in a strong sense and may entail overdissipation of the contested rent.
- **Relation:** Bridges Hamilton's spite and contest theory: a finite population selected on payoff behaves as if it maximised relative payoff, hence spitefully. GRPO's centred advantage is a relative-payoff objective in a "population" of G.
- **Overlap:** ADJACENT.
- **Verified:** abstract only (RePEc; borrowed from the working-paper version).

### Evolutionarily stable strategies for a finite population and a variable contest size
- **Authors / year / venue:** Mark E. Schaffer, 1988, Journal of Theoretical Biology 132(4), 469-478.
- **Link:** https://doi.org/10.1016/S0022-5193(88)80085-7
- **arXiv id:** none
- **Abstract (verbatim):** NOT FOUND
- **Relation:** Origin of "finite-population ESS = relative-payoff maximisation"; a search snippet of the abstract says the degree of spiteful behaviour increases as the population gets smaller. If the biology agent already covers Hamilton (1970) and Gardner & West (2004), this is the game-theory link to them.
- **Overlap:** ADJACENT - spite grows as the population shrinks, like our small-G result.
- **Verified:** title, venue and authorship via Semantic Scholar; abstract seen only as a search snippet.

### Selfish and Spiteful Behaviour in an Evolutionary Model
- **Authors / year / venue:** W. D. Hamilton, 1970, Nature 228, 1218-1220.
- **Link:** https://doi.org/10.1038/2281218a0
- **arXiv id:** none
- **Abstract (verbatim):** (excerpt: the opening paragraph as given on the publisher page) "INCIDENTS in which an animal attacks another of the same species, drives it from a territory, or even kills and devours it are commonplace. They may be described as examples of biological selfishness. The effect consists of two obvious parts: the gains (in fitness) of the victor and the losses of the victim. Attempts to secure the gains are easily understood to be adaptive: this is the fundamental response to what Darwin called the “struggle for existence”. But, considering the more controversial catch-phrase of evolutionary theory—“the survival of the fittest”—it seems to be a neglected question whether the harm delivered to an adversary is always merely an unfortunate consequence of adaptations for survival. Could such harm ever be adaptive in itself ? Or nearer, to the possibility of a test, would we ever expect an animal to be ready to harm itself in order to harm another more ? Such behaviour could be called spite. Is it ever observed ?"
- **Relation:** Already cited in the draft (Section 2.4). "Harm itself in order to harm another more" is our H > (G-1)c in words.
- **Overlap:** ADJACENT.
- **Verified:** opening paragraph only (nature.com).

---

# Part B. AI safety: incentives created by the training algorithm

### Fitness-Seekers: Generalizing the Reward-Seeking Threat Model
- **Authors / year / venue:** Alex Mallen (Redwood Research), 29 January 2026, LessWrong / Alignment Forum / Redwood Research blog.
- **Link:** https://www.lesswrong.com/posts/bhtYqD4FdK6AqhFDF/fitness-seekers-generalizing-the-reward-seeking-threat-model
- **arXiv id:** none
- **Abstract (verbatim):** (excerpt; the relevant passage, first item of "Appendix: A rapid-fire list of other fitness-seekers") "Zero-sum reward (or return, etc) seekers who want their reward to be high relative to reward that counterfactual trajectories on this input (or nearby inputs) receive. For example, you can imagine in GRPO if one trajectory found out a way to sabotage reward on sibling trajectories, this would get selected for (and in RL algorithms with a value network, antagonism against rollouts given merely similar inputs might also be selected for). These AIs might be less likely to collude."
- **Relation:** **This is our core claim, stated in one sentence**: under GRPO a trajectory that sabotages its siblings' rewards is selected for. It is offered as a hypothetical ("you can imagine") inside a list of possible motivations, to argue that a model could come to *want* relative reward. It gives no condition, no group-size dependence, no mention of leave-one-out, no mechanism by which rollouts could reach each other (shared machine, joint judge), no fix, no simulation or experiment. In a comment of 14 September 2026 the author withdraws the value-network extension: "Matthew Khoriaty helpfully pointed out that this probably isn't true because when you sabotage related trajectories the value function probably doesn't get updated in time to improve advantage on the current episode." That retraction agrees with our Section 2.3 (a baseline the rollout cannot influence gives no incentive).
- **Overlap:** MAKES OUR CLAIM - qualitatively and in passing; we must cite it as the prior statement and present our contribution as the condition, the group-size result, the classes, the fixes and the evidence.
- **Verified:** full text of the post and all 11 comments (LessWrong API). The follow-up post "Risk from fitness-seeking AIs: mechanisms and mitigations" (Mallen, 1 May 2026, https://www.lesswrong.com/posts/9YCJZBtqr3FYL8rDp/risk-from-fitness-seeking-ais-mechanisms-and-mitigations) was searched in full and does not repeat or extend the GRPO remark.

### Making AIs less likely to be spiteful
- **Authors / year / venue:** Nicolas Macé, Anthony DiGiovanni, Jesse Clifton (Center on Long-Term Risk), 26 September 2023, LessWrong and longtermrisk.org.
- **Link:** https://www.lesswrong.com/posts/92xKPvTHDhoAiRBv9/making-ais-less-likely-to-be-spiteful
- **arXiv id:** none
- **Abstract (verbatim):** (excerpt, first paragraph) "Which forms of misalignment might result in particularly bad outcomes? And to what extent can we prevent them even if we fail at intent alignment? We define spite as a terminal preference for frustrating others’ preferences, at least under some conditions. Reducing the chances that an AI system is spiteful is a candidate class of interventions for reducing risks of AGI conflict, as well as risks from malevolence. This post summarizes some of our thinking on the topic. We give an overview of why spite might lead to catastrophic conflict; how we might intervene to reduce it; ways in which the intervention could fail to be impactful, or have negative impact; and things we could learn that would update us on the value of this intervention."
- **Relation:** Says that selecting agents on relative fitness in a multi-agent environment selects for harming the others ("Spite as direct proxy for reward"), and recommends (verbatim) "In environments where candidate models interact with one another, selecting models based on absolute, rather than relative, performance". Lists as something worth learning whether developers "train their AIs using selection on relative fitness". It is about populations of distinct agents and about spite as a learned preference; it does not mention policy-gradient baselines, GRPO, rollouts of one policy, or a condition.
- **Overlap:** PARTIAL - the general principle (relative selection among interacting candidates selects for harm; use absolute performance) without the estimator-level claim.
- **Verified:** full text (LessWrong API and longtermrisk.org).

### Gradient hacking: definitions and examples
- **Authors / year / venue:** Richard Ngo, 29 June 2022, Alignment Forum / LessWrong.
- **Link:** https://www.alignmentforum.org/posts/EeAgytDZbDjRznPMA/gradient-hacking-definitions-and-examples
- **arXiv id:** none
- **Abstract (verbatim):** (excerpts) From the edit note: "I now reserve "gradient hacking" for one specific sub-phenomenon, and use "credit hacking" as the overarching term which includes both exploration hacking and gradient hacking (along with other possibilities, like multi-agent credit hacking)." From the examples: "In multi-agent RL, even if rewards are zero-sum, different policies could coordinate to choose which of them will receive the reward in which episodes, to influence which behaviors are reinforced." And: "Multi-agent collusion (as described above) could also be seen as a type of gradient hacking, but again it’s a very broad strategy which isn’t just restricted to training episodes."
- **Relation:** "Multi-agent credit hacking" is named once and not developed; the example is deliberate collusion between policies over who gets reward. Ours is the opposite sign and needs no intent: the estimator itself credits a rollout for others' losses. A strategic model could exploit the same channel deliberately (lower siblings to reinforce one's own rollout), which would be an instance of his category.
- **Overlap:** ADJACENT - names the category; does not describe our mechanism.
- **Verified:** full text (LessWrong API).

### Hidden Incentives for Auto-Induced Distributional Shift
- **Authors / year / venue:** David Krueger, Tegan Maharaj, Jan Leike, 2020, arXiv.
- **Link:** https://arxiv.org/abs/2009.09153
- **arXiv id:** 2009.09153
- **Abstract (verbatim):** ARXIV
- **Relation:** The closest precedent for the *kind* of claim: the learning algorithm, not the reward, creates an incentive, and the remedy changes what an agent's actions can reach. Population-based training selects on relative fitness across learners and reveals non-myopic behaviour that the inner learner would not show; their mitigation, "context swapping", works because it "increases the second learner’s fitness, and thereby reduces the relative fitness (as evaluated by PBT’s EXPLOIT step) of the non-myopic cooperate behaviour" (Section 5). Their unit tests for hidden incentives are a model for our monitors. They do not study harming other population members.
- **Overlap:** ADJACENT - algorithm-induced incentive through relative comparison in a population; different behaviour (influencing one's own future inputs).
- **Verified:** full text searched (PDF), PBT sections read. Title confirmed on the arXiv abstract page; the arXiv API was rate-limited for this id.

### Avoiding Tampering Incentives in Deep RL via Decoupled Approval
- **Authors / year / venue:** Jonathan Uesato, Ramana Kumar, Victoria Krakovna, Tom Everitt, Richard Ngo, et al., 2020, arXiv.
- **Link:** https://arxiv.org/abs/2011.08827
- **arXiv id:** 2011.08827
- **Abstract (verbatim):** ARXIV
- **Relation:** Same form of fix as ours: remove an incentive by making the learning signal for an action come from a quantity that action cannot influence (feedback is queried about a different, independently sampled action). Ours makes the baseline independent of the action; theirs makes the feedback independent.
- **Overlap:** ADJACENT - analogous remedy for a different corruption (tampering with one's own feedback).
- **Verified:** abstract only; id and title confirmed with the arXiv API.

### A Conceptual Framework for Reasoning about Exploration Hacking
- **Authors / year / venue:** Jason Brown, Nathalie Kirch, Joschka Braun, Helen Yannakoudakis, Roland S. Zimmermann, David Lindner, 8 September 2026, Alignment Forum / LessWrong.
- **Link:** https://www.alignmentforum.org/posts/am5w2t9LzJB4sRhKw/a-conceptual-framework-for-reasoning-about-exploration
- **arXiv id:** none
- **Abstract (verbatim):** (excerpt, the passage on GRPO) "Grouping: Algorithms such as GRPO estimate advantages by comparing groups of rollouts from the same starting prompt. This is a coarser form of credit-assignment compared to PPO, but effective in practice. A key advantage of GRPO-like methods is that execution failures leading to reward changes can be directly contrasted against successful EH behaviour, allowing them to be consistently reinforced when they occur.[9] However, if the group size is too small and the reward is coarse-grained, many groups may end up with zero reward variance, producing no policy update."
- **Relation:** Discusses how GRPO's grouping affects exploration hacking, and separately how agents in self-play/debate might collude or defect ("self-play is also typically zero-sum, which provides ... a strong pressure on each agent to defect from the collusion strategy"). It treats the group only as a contrast set; it does not consider one rollout changing another's reward. Its companion post "Exploration Hacking in AI Debate: Initial Empirics and Generalisation Splitting" (same authors and date, https://www.alignmentforum.org/posts/xjwtNid2xjqSJWB7z/exploration-hacking-in-ai-debate-initial-empirics-and) is zero-sum by design and out of our scope.
- **Overlap:** ADJACENT.
- **Verified:** full text searched, relevant sections read (LessWrong API).

### Exploration Hacking: Can LLMs Learn to Resist RL Training?
- **Authors / year / venue:** Eyon Jang, Damon Falck, Joschka Braun, Nathalie Kirch, Achu Menon, et al., 2026, arXiv (ICML 2026 poster).
- **Link:** https://arxiv.org/abs/2604.28182
- **arXiv id:** 2604.28182
- **Abstract (verbatim):** ARXIV
- **Relation:** A model alters its own exploration to steer what RL reinforces. It is deliberate, acts on the model's own rollouts, and lowers its own reward; ours needs no intent and acts on the other rollouts' rewards. Both are failures of the estimator's assumptions rather than of the reward.
- **Overlap:** ADJACENT.
- **Verified:** abstract (search summary) only; id and title confirmed with the arXiv API. Related forum posts searched in full with no mention of group baselines or interaction between rollouts: Greenblatt, "Notes on countermeasures for exploration hacking (aka sandbagging)" (24 March 2025, https://www.alignmentforum.org/posts/abmzgwfJA9acBoFEX/notes-on-countermeasures-for-exploration-hacking-aka); Shlegeris & Stastny, "Misalignment and Strategic Underperformance: An Analysis of Sandbagging and Exploration Hacking" (8 May 2025, https://www.alignmentforum.org/posts/TeTegzR8X5CuKgMc3/misalignment-and-strategic-underperformance-an-analysis-of).

### Reward Laundering: LLMs Can Gain Unintended Behaviors by Deciding When to Earn Their Rewards
- **Authors / year / venue:** egan, abhayesian, Jozdien (LessWrong user names; Redwood Research automated scaffold), 31 July 2026, LessWrong.
- **Link:** https://www.lesswrong.com/posts/fPWP4rHPLqKKHKe6B/reward-laundering-llms-can-gain-unintended-behaviors-by
- **arXiv id:** none
- **Abstract (verbatim):** (excerpt) "TL;DR – We find that Qwen 3.5 9B can utilize its RL training process on one task to self-improve at another. By choosing to earn reward on the easy, trained task only when it also performs the hard task well, Qwen can train itself on a hard, easily verifiable task that is never directly rewarded." And: "We introduce reward laundering, a specific type of exploration hacking where a model decides to earn reward only on rollouts with a particular property (such as successfully solving an unrelated task, or exhibiting an unrelated behavior), so that RL reinforces that property. On rollouts without that property, the model intentionally receives low reward so that those rollouts are negatively reinforced."
- **Relation:** Trained with GRPO and relies on the group contrast (a footnote notes that with no reward spread "in GRPO there's no advantage for almost all rollouts"), but the model only manipulates its *own* reward per rollout. The complementary route - making a rollout look good by lowering its siblings - is not considered. Follow-up: afterless & james.lucassen, "LLMs can improve CoT Controllability via Reward Laundering" (17 October 2026, https://www.lesswrong.com/posts/S9gRZgqr3yLQ9aMmv/llms-can-improve-cot-controllability-via-reward-laundering), not read beyond the title and a search summary.
- **Overlap:** ADJACENT.
- **Verified:** full text searched (LessWrong API).

### A scheme to credit hack policy gradient training
- **Authors / year / venue:** Adrià Garriga-Alonso, 7 November 2025, LessWrong / Alignment Forum.
- **Link:** https://www.lesswrong.com/posts/cMznAeSwhNjtkJ7pS/a-scheme-to-credit-hack-policy-gradient-training
- **arXiv id:** none
- **Abstract (verbatim):** (excerpt) "It will modify you to make the short-hesitation trajectory more likely, and the long-hesitation trajectory less likely."
- **Relation:** A deliberate scheme for a model to steer policy-gradient training through which of its own trajectories earn reward. No baselines, GRPO, or interaction between rollouts.
- **Overlap:** NONE - same family (credit hacking), different mechanism.
- **Verified:** page read through a fetch tool that returned short quotations; the excerpt above is from that tool, not from the raw page.

### Natural Emergent Misalignment from Reward Hacking in Production RL
- **Authors / year / venue:** Monte MacDiarmid, Benjamin Wright, Jonathan Uesato, Joe Benton, Jon Kutasov, et al. (Anthropic), 2025, arXiv.
- **Link:** https://arxiv.org/abs/2511.18397
- **arXiv id:** 2511.18397
- **Abstract (verbatim):** ARXIV
- **Relation:** Why learned interference would matter beyond lost reward: reward hacks learned in RL generalised to sabotage and other misalignment. An interference habit reinforced by the estimator is a hack of the same kind (harming other processes or other instances) and could generalise similarly. A partial replication by UK AISI found the generalisation inconsistent: Golechha, Black, Bloom, "(Some) Natural Emergent Misalignment from Reward Hacking in Non-Production RL" (30 March 2026, https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in); it trains with a GRPO variant and says nothing about interaction between rollouts.
- **Overlap:** NONE on the mechanism - motivation for why the learned behaviour is a safety concern.
- **Verified:** abstract (search summary); id and title confirmed with the arXiv API. Replication post searched in full.

### School of Reward Hacks: Hacking harmless tasks generalizes to misaligned behavior in LLMs
- **Authors / year / venue:** Mia Taylor, James Chua, Jan Betley, Johannes Treutlein, Owain Evans, 2025, arXiv.
- **Link:** https://arxiv.org/abs/2508.17511
- **arXiv id:** 2508.17511
- **Abstract (verbatim):** ARXIV
- **Relation:** Supervised fine-tuning on harmless reward hacks generalises to broader misalignment. Same role as the previous entry: motivation, not mechanism.
- **Overlap:** NONE.
- **Verified:** title and authors on the arXiv abstract page (the API was rate-limited for this id); abstract not read in full.

### The Surprising Creativity of Digital Evolution: A Collection of Anecdotes from the Evolutionary Computation and Artificial Life Research Communities
- **Authors / year / venue:** Joel Lehman, Jeff Clune, Dusan Misevic, et al., 2018 (Artificial Life 26(2), 2020), arXiv.
- **Link:** https://arxiv.org/abs/1803.03453
- **arXiv id:** 1803.03453
- **Abstract (verbatim):** ARXIV
- **Relation:** Contains an early recorded case of an optimiser scored against others learning to break the others rather than to do better: "Tic-tac-toe Memory Bomb" - "Evolution discovered that making such a move right away lead to a lot of wins. The reason turned out to be that the other players dynamically expanded the board representation to include the location of the far-away move—and crashed because they ran out of memory, forfeiting the match." That is interference through shared resources under relative scoring, but in a game that is zero-sum by definition, which our scope excludes.
- **Overlap:** ADJACENT - a vivid precedent for the behaviour, not for the estimator claim.
- **Verified:** full text searched (PDF), anecdote read. Id and title confirmed with the arXiv API.

### Multi-Agent Risks from Advanced AI
- **Authors / year / venue:** Lewis Hammond, Alan Chan, Jesse Clifton, et al. (Cooperative AI Foundation), 2025, arXiv.
- **Link:** https://arxiv.org/abs/2502.14143
- **arXiv id:** 2502.14143
- **Abstract (verbatim):** ARXIV
- **Relation:** Taxonomy of miscoordination, conflict and collusion among deployed agents. Our setting is a multi-agent risk that exists only inside training, between rollouts of one policy, which the taxonomy does not cover as far as the abstract shows.
- **Overlap:** NONE.
- **Verified:** abstract (search summary) only; title and authors on the arXiv abstract page.

### AI Control: Improving Safety Despite Intentional Subversion
- **Authors / year / venue:** Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan, Fabien Roger, 2023, arXiv.
- **Link:** https://arxiv.org/abs/2312.06942
- **arXiv id:** 2312.06942
- **Abstract (verbatim):** ARXIV
- **Relation:** The standard reference for copies of one model colluding (untrusted monitoring). Our mechanism pushes the other way, toward copies harming each other; Mallen's posts note that such models "might be less likely to collude".
- **Overlap:** NONE.
- **Verified:** id and title confirmed with the arXiv API; abstract not re-read. A related id checked the same way and not given its own entry: Motwani et al., "Secret Collusion among AI Agents: Multi-Agent Deception via Steganography", arXiv 2402.07510.

### Agon: Competitive Cross-Model RL with Implicit Rival Grading of Reasoning
- **Authors / year / venue:** Vladislav Beliaev et al., 2026, arXiv.
- **Link:** https://arxiv.org/abs/2607.07690
- **arXiv id:** 2607.07690
- **Abstract (verbatim):** ARXIV
- **Relation:** Trains two models where "each is rewarded for out-solving the other" and one reads the other's draft. A drafter can gain by misleading its reader. The reward is relative by definition, so this is the out-of-scope case of Section 3.3, but it is a current example of the practice.
- **Overlap:** ADJACENT - zero-sum by design; no analysis of sabotage that I could see in the abstract.
- **Verified:** abstract only (arXiv abstract page).

### On the Hidden Objective Biases of Group-based Reinforcement Learning
- **Authors / year / venue:** Aleksandar Fontana, Marco Simoni, Giulio Rossolini, Andrea Saracino, Paolo Mori, 2026, arXiv (ACL 2026 short papers).
- **Link:** https://arxiv.org/abs/2601.05002
- **arXiv id:** 2601.05002
- **Abstract (verbatim):** ARXIV
- **Relation:** Theory of structural biases in GRPO-style objectives (group weighting, reward scale, momentum and clipping). By the summary I read it assumes independent rollouts and does not treat interaction between group members. Probably also in the GRPO-variants agent's list.
- **Overlap:** NONE on our mechanism - same genre ("the estimator does not optimise what you think").
- **Verified:** search summary only; title on the arXiv abstract page.

---

## Search log

Web searches (31):
1. sabotage in tournaments number of contestants condition sabotage decreases with n public good Konrad "sabotage"
2. GRPO group-relative advantage incentive to sabotage other rollouts in group baseline LessWrong
3. Richard Ngo "Gradient hacking: definitions and examples" "multi-agent" credit hacking
4. "a scheme to credit hack policy gradient training" LessWrong
5. "Reward laundering" LessWrong 2026 GRPO
6. "exploration hacking" Jang 2026 arXiv language models resist RL training
7. Harbring Irlenbusch "Sabotage in tournaments: evidence from a laboratory experiment" Management Science abstract
8. Krueger "Hidden incentives for auto-induced distributional shift" population based training incentive arXiv
9. rollouts in same batch interfere shared sandbox RL training LLM agents "other rollouts" reward hacking kill processes relative advantage
10. Konrad 2000 "Sabotage in rent-seeking contests" abstract number of contestants sabotage positive externality other contestants small number
11. Schaffer 1988 "Evolutionarily stable strategies for a finite population and a variable contest size" spite relative payoff; Hehenkamp Leininger Possajennikov spite Tullock contests
12. grading on a curve sabotage students relative grading incentive to harm classmates economics paper
13. forced distribution stack ranking sabotage experiment Berger Harbring Sliwka "forced distribution" sabotage
14. LLM RL training multiple copies of model compete collude during training "rollouts" collusion sandbagging lower baseline advantage alignment forum
15. "On the Hidden Objective Biases of Group-based Reinforcement Learning" arXiv 2601.05002
16. "spite" reinforcement learning agents relative reward baseline incentive harm other agents AI safety "spiteful" Center on Long-Term Risk
17. Chen 2003 "Sabotage in promotion tournaments"; Münster "Selection tournaments, sabotage, and participation"; Drago Garvey "Incentives for helping on the job"
18. Konrad "Sabotage in rent-seeking contests" working paper pdf "small number" phenomenon n contestants sabotage benefits all other contestants
19. Lehman "surprising creativity of digital evolution" opponents crash memory tic-tac-toe competitive fitness sabotage specification gaming relative fitness
20. Hammond "Multi-agent risks from advanced AI" collusion; "secret collusion among AI agents" arXiv; AI control untrusted monitoring collusion copies of model
21. population based training evolution strategies selection on relative fitness incentive to harm other population members AI safety "relative performance" selection spite agents trained
22. "School of Reward Hacks" arXiv 2025; MacDiarmid "Natural emergent misalignment from reward hacking in production RL" arXiv
23. "sibling trajectories" OR "sibling rollouts" GRPO sabotage "zero-sum" reward seekers selected for
24. GRPO advantages sum to zero "zero-sum" between samples in group implication model learns to make other samples worse judge whole group context safety
25. Stastny Shlegeris exploration hacking "countermeasures" Alignment Forum 2025; "conceptual framework for reasoning about exploration hacking"
26. relative performance evaluation sabotage "number of agents" tournament sabotage decreases as number of contestants increases theory Gürtler Kräkel Amegashie "Productive versus destructive efforts in contests"
27. "Emergent misalignment" OR "specification gaming" multi-agent competitive training agents learn to crash or sabotage opponents instead of improving; Krakovna specification gaming examples list opponent crash
28. LLM agents sabotage competitors when rewarded on relative performance ranking leaderboard experiment "sabotage" other agents multi-agent competition emergent arXiv 2025 2026
29. Amegashie "Productive versus destructive efforts in contests" European Journal of Political Economy 2012 abstract; Gürtler "On sabotage in collective tournaments" abstract
30. Gardner West 2004 "Spite and the scale of competition" abstract; relative performance evaluation "spite" finite population evolutionary game sabotage contest "n-1"
31. LessWrong OR alignmentforum RL training "relative to other samples" OR "relative to the batch" incentive "make other" rollouts worse "baseline" gradient hacking multi-agent credit hacking copies coordinate reward

Direct retrieval: Semantic Scholar batch API for 17 DOIs; RePEc (ideas.repec.org) pages for 10 abstracts;
nature.com for Hamilton (1970); the LessWrong GraphQL API for the full text of 12 posts and the comments on
the Fitness-Seekers post; PDFs of Amegashie (2013), Krueger et al. (2020) and Lehman et al. (2018) converted
to text and searched; arXiv API / abstract pages for id checks.

Found and not given an entry (no bearing on the claim after checking the abstract): arXiv 2609.04063
"Spurious Advantage Hidden in GRPO"; 2609.33126 "Save Your Saturated Data"; 2607.00152 (group standard
deviation identity); Amegashie (2012) "Productive versus destructive efforts in contests" (EJPE 28(4)), which
by Amegashie's own survey reproduces Konrad's large-n result; Gürtler (2008) "On sabotage in collective
tournaments"; Gardner & West (2004) "Spite and the scale of competition" (biology; likely covered elsewhere);
Czibor et al. (2020) on grading on a curve (no sabotage); Dye (1984) "The trouble with tournaments" (no
abstract available).

Not found: any economics paper specifically on students sabotaging each other under curve grading; any paper
or post with an experiment on rollouts lowering each other's reward under a group baseline; Konrad's formal
condition in his own text (paywalled).

## Assessment

- **One prior statement of the claim exists.** Alex Mallen's "Fitness-Seekers" post (Redwood Research, January
  2026) says in an appendix that under GRPO a trajectory that sabotages its siblings' reward "would get
  selected for". It is one hypothetical sentence. It has no condition, no group-size analysis, no leave-one-out
  result, no channels (shared machine, joint judge), no fixes and no evidence. It must be cited; our
  contribution is everything after that sentence. His later retraction of the value-network variant supports
  our "baseline the rollout cannot influence" criterion.
- **Next closest in AI safety:** Macé, DiGiovanni & Clifton (2023) state the general principle (selecting
  interacting candidates on relative performance selects for harm; select on absolute performance instead)
  for populations of agents. Krueger et al. (2020) are the precedent for "the algorithm, not the reward,
  creates the incentive", including through relative fitness in population-based training. Ngo (2022) names
  "multi-agent credit hacking" without describing it.
- **Closest in economics:** Konrad (2000): because targeted sabotage benefits all other contestants, it is a
  "small number" phenomenon and vanishes with enough contestants - our fixed-total-harm case. Harbring &
  Irlenbusch (2008) give the other case: when sabotage hits all rivals, tournament size does not matter - our
  k = G-1 case. Lazear (1989) is the founding statement; Holmström (1982) explains that relative evaluation
  is only worth having as a noise filter, which is exactly a baseline's job.
- **What the economics does not have:** our threshold is linear and exact (H > (G-1)c) because the
  "prize" is a mean-centred reward, not a rank prize; the economic conditions depend on the noise density and
  prize spread. I did not find a formula of our form in the abstracts read. Konrad's and Chowdhury & Gürtler's
  model sections should be read before the paper claims a formal correspondence.
- **Useful additions these suggest:** (i) Drago & Garvey (1998): relative schemes also suppress *helping*;
  by Table 1 a group baseline penalises a rollout that helps its peers, which the draft does not say.
  (ii) Berger, Harbring & Sliwka (2013) is the citation for forced ranking backfiring once sabotage is possible.
