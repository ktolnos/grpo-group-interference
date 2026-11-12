# Related work 04: multi-agent credit assignment, relative rewards, and spite

Search date: 11 November 2026. Topic: difference rewards / COIN / COMA; relative or rank-based
rewards and sabotage; evolution of spite in finite populations; relative fitness in
evolutionary computation and population-based training.

All 22 arXiv ids below were checked against the arXiv API (title matched). Venues marked "not checked" were not verified; only the arXiv record was. Abstracts of
non-arXiv works were copied from PubMed, Crossref, RePEc or the paper's own PDF, as stated
in each entry. "NOT FOUND" means no exact text could be retrieved.

## Summary of the spite condition in the literature (read this first)

Our condition: an act that costs the actor c and does total harm D to the other G − 1
rollouts is reinforced by GRPO / leave-one-out when **D > (G − 1)c**.

**Hamilton (1970), as restated by Patel, West & Biernaskie (2019, Eq. 5).** Read in the
bioRxiv PDF. For a well-mixed population of N in which the actor harms others
indiscriminately, harming is favoured when

  −c + (−1/(N − 1)) · b1 > 0,

where, quoting the paper, "–1/(N – 1) is the relatedness between actor and primary
recipients, measured with respect to the entire population", and "This is the version of
negative relatedness that has led to predictions of indiscriminate spite in small
populations (e.g., Hamilton 1971; Grafen 1985)." Here c is the cost to the actor and b1 the
fitness effect on the harmed (primary) recipients, negative for harm. Writing b1 = −D gives
**D > (N − 1)c**, the same inequality as ours with N = G.
*Caveat:* we read this in Patel et al., not in Hamilton's own letter (paywalled). Whether
b1 is the total effect on all harmed recipients or a per-recipient effect should be checked
against Patel et al.'s Eq. 1–4 before we print "identical"; the form of the inequality and
the −1/(N − 1) weight are as quoted.

Why the two agree: GRPO's advantage is A_i = (1 − 1/G)·(r_i − mean of the others), so
another rollout's reward enters with weight −1/(G − 1) relative to one's own. That weight
is exactly Hamilton's finite-population relatedness to a random other member, measured
against a population mean that includes the actor.

**Schaffer (1988), finite-population ESS.** We could not obtain the full text. The
definition is restated in Duersch, Oechssler & Schipper (2010, Def. 8, read in full):
x* is a finite-population ESS if π(x*, x) ≥ π(x, x*) for all x, i.e. a single mutant must
not earn more than the incumbents it plays against; and their Proposition 2 (attributed to
Schaffer 1988, 1989) states that this is equivalent to x* being a saddle point of the
*relative payoff game* Δ(x, y) = π(x, y) − π(y, x). Neill (2004, read in full) gives the
mechanism in words: "In a finite population, the probability that a mutant plays against
another mutant is smaller than the probability that a player of the common strategy plays
against a mutant, since the mutant player cannot face itself in a contest."

*Our derivation from that definition (not a quotation from Schaffer):* one mutant in a
population of N, incumbents play s.
- *Pairwise contests, random matching.* The mutant always meets an incumbent and earns
  π(m, s). An incumbent meets the mutant with probability 1/(N − 1) and earns
  [(N − 2)/(N − 1)]·π(s, s) + [1/(N − 1)]·π(s, m). Let c = π(s, s) − π(m, s) (cost to the
  mutant) and d = π(s, s) − π(s, m) (harm to its opponent). The mutant out-earns the
  incumbents, and so invades, when **d > (N − 1)c**. This is our fixed-total-harm case
  (one victim per act, D = d).
- *Playing the field (contest size N).* The mutant harms each of the N − 1 others by d.
  It out-earns them when **d > c**, independent of N. This is our case k = G − 1.

So Schaffer's invasion criterion "mutant payoff minus incumbent payoff > 0" is the
leave-one-out advantage r_i − mean of the others, and our Table 1 rows for GRPO and RLOO
reproduce his finite-population result. Schaffer's abstract states the qualitative
conclusion: the degree of spite "is shown to increase with a decrease in the population
size".

**A warning from the same literature.** Patel et al. argue that this behaviour should be
called *selfish*, not spiteful, because the actor "gains a direct feedback benefit from
harming others" through reduced competition. In our setting the feedback benefit is the
lowered baseline. A referee from that field may object to the word "spite"; "Hamilton's
(1970) finite-population condition" is safer than "Hamilton's spite".

---

## A. Evolution of spite and finite-population relative fitness

### Selfish and Spiteful Behaviour in an Evolutionary Model
- **Authors / year / venue:** W. D. Hamilton, 1970, Nature 228(5277): 1218–1220.
- **Link:** https://doi.org/10.1038/2281218a0
- **arXiv id:** none
- **Abstract (verbatim):** The letter has no abstract. Opening paragraph, as shown on the Nature article page (excerpt): "INCIDENTS in which an animal attacks another of the same species, drives it from a territory, or even kills and devours it are commonplace. They may be described as examples of biological selfishness. The effect consists of two obvious parts: the gains (in fitness) of the victor and the losses of the victim. Attempts to secure the gains are easily understood to be adaptive: this is the fundamental response to what Darwin called the “struggle for existence”. But, considering the more controversial catch-phrase of evolutionary theory—“the survival of the fittest”—it seems to be a neglected question whether the harm delivered to an adversary is always merely an unfortunate consequence of adaptations for survival. Could such harm ever be adaptive in itself ? Or nearer, to the possibility of a test, would we ever expect an animal to be ready to harm itself in order to harm another more ? Such behaviour could be called spite. Is it ever observed ?"
- **Relation:** Source of the idea that, when similarity is measured against the whole finite population including the actor, relatedness to others is negative (−1/(N − 1) as restated by Patel et al.), so costly harm can be selected in small populations. Our group-size result has the same form.
- **Overlap:** ADJACENT — same mathematical structure in population genetics; nothing about learning algorithms or baselines.
- **Verified:** Nature landing page (first paragraph) and PubMed record only; the condition itself was read in Patel et al. (2019), not in the letter.

### An evolutionarily stable strategy approach to indiscriminate spite
- **Authors / year / venue:** N. Knowlton, G. A. Parker, 1979, Nature 279(5712): 419–421.
- **Link:** https://doi.org/10.1038/279419a0
- **arXiv id:** none
- **Abstract (verbatim):** "AN individual behaves spitefully when it harms itself in order to harm another individual more(1). Hamilton(1,2) predicted that spite may evolve if it is expressed only in those encounters that occur between individuals of less than average relatedness. More recently Verner(3) suggested that territory size may become super-optimal because of a selective advantage arising from the spiteful exclusion of others from limited resources. His model is essentially different from Hamilton's in that spite is directed at individuals indiscriminately with respect to relatedness. Recently Rothstein(4) has shown analytically that the initial spread of spiteful traits will be very slow in all but the smallest populations. He also argued verbally that indiscriminate spite can never be evolutionarily stable even if it should spread (see also Davies(5)). The question of evolutionary stability is clearly important, but its resolution requires an analytical approach. We report here an approach based on Maynard Smith's(6) concept of the evolutionarily stable strategy (ESS), a strategy which, when common, does better than any alternative strategy played by a rare mutant. We show that spite can be an ESS, but that the magnitude of spite will be small in large populations."
- **Relation:** Indiscriminate spite (harm to anyone, as in our setting where a rollout harms whichever siblings share its machine) is an ESS whose size shrinks with population size. Matches our fixed-total-harm group-size dependence.
- **Overlap:** ADJACENT — same qualitative result in biology.
- **Verified:** abstract only (PubMed).

### Evolutionarily stable strategies for a finite population and a variable contest size
- **Authors / year / venue:** M. E. Schaffer, 1988, Journal of Theoretical Biology 132(4): 469–478.
- **Link:** https://doi.org/10.1016/S0022-5193(88)80085-7
- **arXiv id:** none
- **Abstract (verbatim):** "This paper presents a generalization of Maynard Smith's concept of an evolutionarily stable strategy (ESS) to cover the cases of a finite population and a variable contest size. Both equilibrium and stability conditions are analysed. The standard Maynard Smith ESS with an infinite population and a contest size of two (pairwise contests) is shown to be a special case of this generalized ESS. An important implication of the generalized ESS is that in finite populations the behaviour of an ESS player is "spiteful", in the sense that an ESS player acts not only to increase his payoff but also to decrease the payoffs of his competitors. The degree of this "spiteful" behaviour is shown to increase with a decrease in the population size, and so is most likely to be observed in small populations. The paper concludes with an extended example: a symmetric two-pure-strategies two-player game for a finite population. It is shown that a mixed strategy ESS is globally stable against invasion by any one type of mutant strategist. The condition for the start of simultaneous invasion by two types of mutant is also given."
- **Relation:** The closest formal ancestor of our Section 2. Selection on payoff relative to the rest of a finite population makes lowering others' payoffs count like raising one's own, more strongly in smaller populations. "Variable contest size" corresponds to our k (how many peers one act reaches); see the derivation at the top of this file.
- **Overlap:** PARTIAL — the same inequality follows from his definition, but for natural selection on relative payoff, not for a policy-gradient estimator whose baseline is meant to be a neutral variance-reduction device.
- **Verified:** abstract only (PubMed). Definition and Proposition read second-hand in Duersch et al. (2010) and Neill (2004). The full text must be read before we cite a specific equation.

### Are profit-maximisers the best survivors? A Darwinian model of economic natural selection
- **Authors / year / venue:** M. E. Schaffer, 1989, Journal of Economic Behavior & Organization 12(1): 29–45.
- **Link:** https://doi.org/10.1016/0167-2681(89)90075-9
- **arXiv id:** none
- **Abstract (verbatim):** NOT FOUND
- **Relation:** Applies the finite-population ESS to firms: survivors maximise relative, not absolute, profit, so they take actions that lower their own profit if rivals' profits fall more. Cited together with Schaffer (1988) for the relative-payoff-game equivalence.
- **Overlap:** ADJACENT — same mechanism in economics.
- **Verified:** existence and bibliographic data (Crossref, Semantic Scholar); content only as described in Duersch et al. (2010).

### Pure Saddle Points and Symmetric Relative Payoff Games
- **Authors / year / venue:** Peter Duersch, Jörg Oechssler, Burkhard C. Schipper, 2010, arXiv preprint (cs.GT).
- **Link:** https://arxiv.org/abs/1003.4277
- **arXiv id:** 1003.4277
- **Abstract (verbatim):** ARXIV
- **Relation:** States Schaffer's finite-population ESS cleanly (Definition 8) and the equivalence with Nash equilibrium of the zero-sum *relative payoff game* Δ(x, y) = π(x, y) − π(y, x) (Proposition 2). GRPO's advantages sum to zero within a group, so GRPO turns any group reward into exactly such a relative payoff game among the rollouts.
- **Overlap:** ADJACENT — gives the game-theoretic name for what GRPO optimises; no learning-algorithm content.
- **Verified:** full text (PDF).

### Evolutionary stability for large populations
- **Authors / year / venue:** Daniel B. Neill, 2004, Journal of Theoretical Biology 227(3): 397–401.
- **Link:** https://doi.org/10.1016/j.jtbi.2003.11.017
- **arXiv id:** none
- **Abstract (verbatim):** "We present a revision of Maynard Smith's evolutionary stability criteria for populations which are very large (though technically finite) and of unknown size. We call this the large population ESS, as distinct from Maynard Smith's infinite population ESS and Schaffer's finite population ESS. Building on Schaffer's finite population model, we define the large population ESS as a strategy which cannot be invaded by any finite number of mutants, as long as the population size is sufficiently large. The large population ESS is not equivalent to the infinite population ESS: we give examples of games in which a large population ESS exists but an infinite population ESS does not, and vice versa. Our main contribution is a simple set of two criteria for a large population ESS, which are similar (but not identical) to those originally proposed by Maynard Smith for infinite populations."
- **Relation:** Shows the finite-population correction does not always vanish as N grows: when the harming act is free to the actor (our c = 0), it is favoured at every finite N. Matches our statement that with c ≤ 0 interference is reinforced at any group size.
- **Overlap:** ADJACENT.
- **Verified:** full text (author's PDF); abstract copied from it (ligatures normalised).

### Spite and the scale of competition
- **Authors / year / venue:** A. Gardner, S. A. West, 2004, Journal of Evolutionary Biology 17(6): 1195–1203.
- **Link:** https://doi.org/10.1111/j.1420-9101.2004.00775.x
- **arXiv id:** none
- **Abstract (verbatim):** "In recent years there has been a large body of theoretical work examining how local competition can reduce and even remove selection for altruism between relatives. However, it is less well appreciated that local competition favours selection for spite, the relatively neglected ugly sister of altruism. Here, we use extensions of social evolution theory that were formulated to deal with the consequences for altruism of competition between social partners, to illustrate several points on the evolution of spite. Specifically, we show that: (i) the conditions for the evolution of spite are less restrictive than previously assumed; (ii) previous models which have demonstrated selection for spite often implicitly assumed local competition; (iii) the scale of competition must be allowed for when distinguishing different forms of spite (Hamiltonian vs. Wilsonian); (iv) local competition can enhance the spread of spiteful greenbeards; and (v) the theory makes testable predictions for how the extent of spite should vary dependent upon population structure and average relatedness."
- **Relation:** The key variable is the *scale of competition*: spite pays when the individuals you harm are the ones you are compared with. In GRPO the scale of competition is the group of G rollouts, by construction; an independent baseline makes competition global and removes the incentive. This is the cleanest biological statement of our fix.
- **Overlap:** ADJACENT — same principle ("who is in your comparison set"), different field.
- **Verified:** abstract only (Crossref / PubMed).

### Spite: altruism's evil twin
- **Authors / year / venue:** W. L. Vickery, J. S. Brown, G. J. FitzGerald, 2003, Oikos 102(2): 413–416.
- **Link:** https://doi.org/10.1034/j.1600-0579.2003.12410.x
- **arXiv id:** none
- **Abstract (verbatim):** NOT FOUND
- **Relation:** Short game-theory model of spite; per a library catalogue summary it finds that the conditions permitting spite parallel those permitting altruism. Cited in the later literature as treating Wilsonian spite.
- **Overlap:** ADJACENT.
- **Verified:** existence and bibliographic data only (Crossref, Semantic Scholar).

### The evolutionary dynamics of spite in finite populations
- **Authors / year / venue:** Rory Smead, Patrick Forber, 2013, Evolution 67(3): 698–707.
- **Link:** https://doi.org/10.1111/j.1558-5646.2012.01831.x
- **arXiv id:** none
- **Abstract (verbatim):** "Spite, the shady relative of altruism, involves paying a fitness cost to inflict a cost on some recipient. Here, we investigate a density dependent dynamic model for the evolution of spite in populations of changing size. We extend the model by introducing a dynamic carrying capacity. Our analysis shows that it is possible for unconditionally spiteful behavior to evolve without population structure in any finite population. In some circumstances spiteful behavior can contribute to its own stability by limiting population growth. We use the model to show that there are differences between spite and altruism, and to refine Hamilton's original argument about the insignificance of spite in the wild. We also discuss the importance of fixing the measure of fitness to classify behaviors as selfish or spiteful."
- **Relation:** Unconditional spite evolves in any finite population, and whether an act counts as "spiteful" or "selfish" depends on whether fitness is measured absolutely or relatively. The same distinction separates our true objective J (absolute) from GRPO's advantage (relative).
- **Overlap:** ADJACENT.
- **Verified:** abstract only (PubMed). Note the DOI ends in 2012.01831.x; the DOI 2011.01466.x that some aggregators attach to this title is a different paper.

### The evolution of fairness through spite
- **Authors / year / venue:** Patrick Forber, Rory Smead, 2014, Proceedings of the Royal Society B 281(1780): 20132439.
- **Link:** https://doi.org/10.1098/rspb.2013.2439
- **arXiv id:** none
- **Abstract (verbatim):** "The presence of apparently irrational fair play in the ultimatum game remains a focal point for studies in the evolution of social behaviour. We investigate the role of negative assortment in the evolution of fair play in the ultimatum game. Spite-social behaviour that inflicts harm with no direct benefit to the actor-can evolve when it is disproportionally directed at individuals playing different strategies. The introduction of negative assortment alters the dynamics in a way that increases the chance fairness evolves, but at a cost: spite also evolves. Fairness is usually linked to cooperation and prosocial behaviour, but this study shows that it may have evolutionary links to harmful antisocial behaviour."
- **Relation:** Spite needs anti-correlated interaction: the actor's harm must fall on others more than on its own type. A rollout never harms itself through the baseline, which is the same anti-correlation (a mutant "cannot face itself").
- **Overlap:** ADJACENT.
- **Verified:** abstract only (PubMed).

### Can natural selection favour indiscriminate spite?
- **Authors / year / venue:** Matishalin Patel, Stuart A. West, Jay M. Biernaskie, 2019, bioRxiv preprint 745638 (a journal version may exist; not checked).
- **Link:** https://doi.org/10.1101/745638
- **arXiv id:** none
- **Abstract (verbatim):** "Spiteful behaviours occur when an actor harms its own fitness to inflict harm on the fitness of others. Several papers have predicted that spite can be favoured in sufficiently small populations, even when the harming behaviour is directed indiscriminately at others. However, it is not clear that truly spiteful behaviour could be favoured without the harm being directed at a subset of social partners with relatively low genetic similarity to the actor (kin discrimination). Using mathematical models, we show that: (1) the evolution of spite requires kin discrimination; (2) previous models suggesting indiscriminate spite involve scenarios where the actor gains a direct feedback benefit from harming others, and so the harming is selfish rather than spiteful; (3) extreme selfishness can be favoured in small populations (and in some cases small groups) because this is where the feedback benefit of harming is greatest."
- **Relation:** Gives the explicit small-population condition −c + (−1/(N − 1))·b1 > 0 (Eq. 5), which is our D > (G − 1)c, and argues the act is really selfish because the actor profits from reduced competition. In GRPO the "reduced competition" is the lowered baseline, so their reading applies directly: GRPO makes interference *selfish with respect to the advantage* although it is harmful with respect to reward.
- **Overlap:** PARTIAL — the same inequality and the same group-size conclusion, in population genetics.
- **Verified:** full text (bioRxiv PDF, text extracted; equations partly garbled by extraction, Eq. 5 confirmed by the in-text restatement "–c + (–1/[N–1])b1").

### Spite is contagious in dynamic networks
- **Authors / year / venue:** Zachary Fulker, Patrick Forber, Rory Smead, Christoph Riedl, 2021, Nature Communications 12.
- **Link:** https://doi.org/10.1038/s41467-020-20436-1
- **arXiv id:** none
- **Abstract (verbatim):** "Spite, costly behavior that harms others, presents an evolutionary puzzle: given that both the actor and recipient do worse, how could it emerge? We show that dynamically evolving interaction networks provide a novel explanation for the evolution of costly harm. Previous work has shown that anti-correlated interaction (e.g., negative assortment or negative relatedness) among behavioral strategies in populations can lead to the evolution of costly harm. We show that these approaches are blind to important features of interaction brought about by a co-evolution of network and behavior and that these features enable the emergence of spite. We analyze a new model in which agents can inflict harm on others at a cost to themselves, and simultaneously learn how to behave and with whom to interact. We find spite emerges reliably under a wide range of conditions. Our model reveals that when interactions occur in dynamic networks the population can exhibit correlated and anti-correlated behavioral interactions simultaneously, something not possible in standard models. In dynamic networks spite evolves due to transient and partial anti-correlated interaction, even when other behaviors are positively correlated and average degree of correlated interaction in the population is low."
- **Relation:** The one spite paper found that uses reinforcement learning (agents reinforce network links by payoff), but behaviours spread by imitation of relatively successful agents; it does not concern policy-gradient baselines.
- **Overlap:** ADJACENT.
- **Verified:** abstract only (Crossref / Semantic Scholar).

### The Evolution of Walrasian Behavior
- **Authors / year / venue:** Fernando Vega-Redondo, 1997, Econometrica 65(2): 375–384.
- **Link:** https://doi.org/10.2307/2171898
- **arXiv id:** none
- **Abstract (verbatim):** As given on IDEAS/RePEc, which notes it "was borrowed from another version of this item" (the working paper): "Consider an evolutionary context where a given number of quantity-setting oligopolists tend to mimic successful behavior, occasionally experimenting with some small probability. In this context, it is shown that the unique long-run outcome of the process has all firms playing Walrasian, i.e., choosing an output that maximizes profits when taking the market-clearing price as given."
- **Relation:** A *learning* rule based on relative success (imitate the best-performing peer) converges to an outcome with lower profit for everyone than Nash play, because over-producing hurts rivals more than oneself. The closest precedent for "an update rule that compares you with your peers drives the group's payoff down".
- **Overlap:** ADJACENT — learning by relative comparison lowers total payoff; imitation dynamics, not policy gradient, and the externality is in the reward (market price), which our paper treats as in scope only because the estimator, not the reward, creates the incentive.
- **Verified:** abstract only (RePEc), and descriptions in Duersch et al. (2010).

### Evolutionary equilibrium in Tullock contests: spite and overdissipation
- **Authors / year / venue:** Burkhard Hehenkamp, Wolfgang Leininger, Alexandre Possajennikov, 2004, European Journal of Political Economy 20(4): 1045–1057.
- **Link:** https://doi.org/10.1016/j.ejpoleco.2003.09.002
- **arXiv id:** none
- **Abstract (verbatim):** NOT FOUND
- **Relation:** Applies Schaffer's finite-population ESS to contests and attributes the extra effort to spite. One of several economics applications of relative-payoff selection.
- **Overlap:** ADJACENT.
- **Verified:** existence and bibliographic data only (Crossref); cited in Duersch et al. (2010).

## B. Relative performance evaluation and sabotage (economics)

### Moral Hazard in Teams
- **Authors / year / venue:** Bengt Holmström, 1982, Bell Journal of Economics 13(2): 324–340.
- **Link:** https://econpapers.repec.org/RePEc:rje:bellje:v:13:y:1982:i:autumn:p:324-340
- **arXiv id:** none
- **Abstract (verbatim):** "This article studies moral hazard with many agents. The focus is on two features that are novel in a multiagent setting: free riding and competition. The free-rider problem implies a new role for the principal: administering incentive schemes that do not balance the budget. This new role is essential for controlling incentives and suggests that firms in which ownership and labor are partly separated will have an advantage over partnerships in which output is distributed among agents. A new characterization of informative (hence valuable) monitoring is derived and applied to analyze the value of relative performance evaluation. It is shown that competition among agents (due to relative evaluations) has merit solely as a device to extract information optimally. Competition per se is worthless. The role of aggregate measures in relative performance evaluation is also explored, and the implications for investment rules are discussed."
- **Relation:** The economics counterpart of a baseline: paying an agent relative to peers is valuable only because it filters out shared noise. That is the reason GRPO subtracts the group mean. Holmström assumes agents cannot affect each other's output.
- **Overlap:** ADJACENT — states the purpose of peer-relative evaluation (variance reduction), not its failure.
- **Verified:** abstract only (RePEc).

### Pay Equality and Industrial Politics
- **Authors / year / venue:** Edward P. Lazear, 1989, Journal of Political Economy 97(3): 561–580.
- **Link:** https://doi.org/10.1086/261616
- **arXiv id:** none
- **Abstract (verbatim):** "Personnel managers often argue that equitable pay treatment manifested as wage compression is useful because it reduces disharmony among workers. But it is far from obvious that a compressed salary structure is morale improving since better workers may feel disenchanted by this scheme. However, when workers' rewards are based on relative comparisons, salary compression reduces uncooperative behavior that is detrimental to the firm. Relative comparisons imply that some reference group must be selected. The major result is that, within the relevant groups, some wage compression is efficient."
- **Relation:** The standard reference for "relative rewards induce sabotage": once pay depends on comparison within a reference group, workers take costly actions that lower co-workers' output. The reference group is our group of G rollouts; "wage compression" corresponds to shrinking the advantage scale. This is the closest non-ML statement of our claim.
- **Overlap:** PARTIAL — same mechanism (peer-relative reward makes harming peers pay, and total output falls) with strategic human agents and an explicitly relative reward; no gradient estimator, no threshold in G.
- **Verified:** abstract only (RePEc); the sabotage model as described in Chowdhury & Gürtler (2015).

### Sabotage in contests: a survey
- **Authors / year / venue:** Subhasish M. Chowdhury, Oliver Gürtler, 2015, Public Choice 164(1): 135–155.
- **Link:** https://doi.org/10.1007/s11127-015-0264-9
- **arXiv id:** none
- **Abstract (verbatim):** "A contest is a situation in which individuals expend irretrievable resources to win valuable prize(s). “Sabotage” is a deliberate and costly act of damaging a rival’s likelihood of winning the contest. Sabotage can be observed in, e.g., sports, war, promotion tournaments, and political or marketing campaigns. In this article, we provide a model of, and different perspectives on, such sabotage activities and review the economics literature analyzing the act of sabotage in contests. We discuss the theories and evidence emphasizing the means of sabotage, why sabotage occurs, and the effects of sabotage on individual contestants and on overall welfare, with possible means of reducing sabotage. We note that most sabotage activities are aimed at the ablest contestant, the possibility of sabotage reduces productive effort exerted by the contestants, and sabotage may lessen the effectiveness of public policies, such as affirmative action, or information revelation in contests. We discuss the different policies a designer might use to counteract sabotage. We conclude by pointing out some areas of future research."
- **Relation:** Survey of the sabotage literature that followed Lazear. One finding in the abstract transfers: sabotage is aimed at the strongest rival (a prediction we could test: do rollouts target the best sibling?). From memory, not from the abstract, the survey also reports that sabotage falls as the number of contestants rises, because harming one rival helps all the others too (our 1/G dilution); check before citing.
- **Overlap:** ADJACENT — contests are zero-sum by definition, which our paper puts out of scope.
- **Verified:** abstract only (RePEc).

## C. Artificial selection with interacting individuals

### Selection in Reference to Biological Groups I. Individual and Group Selection Applied to Populations of Unordered Groups
- **Authors / year / venue:** B. Griffing, 1967, Australian Journal of Biological Sciences 20(1): 127–140.
- **Link:** https://doi.org/10.1071/BI9670127
- **arXiv id:** none
- **Abstract (verbatim):** "A selection theory designed to accommodate interactions among genotypes is presented. This involves defining unordered groups of genotypes among which interactions may occur, and describing populations of groups generated as combinatorial products of base populations. Gene models are developed which consist not only of direct contributions of the genotypes they represent but also of associate effects from other genotypes in the group."
- **Relation:** Splits an individual's effect into a *direct* effect on its own phenotype and *associate* effects on its group mates (our −c and −d), and, as described in secondary sources (Muir 1996 as summarised in search results; not read here), shows that selecting individuals on their own phenotype can lower the population mean when associate effects are negative, whereas selecting whole groups cannot. This is our "group-total credit matches the true gradient" row, in quantitative genetics, 57 years earlier. The second paper in the series (1969, https://doi.org/10.1071/BI9690131) gives an index of direct and associate values that always yields a "maximum, non-negative change in the population mean".
- **Overlap:** PARTIAL — same decomposition and same fix (credit the group total), for breeding rather than gradient estimation.
- **Verified:** abstract only (Crossref).

### Group Selection for Adaptation to Multiple-Hen Cages: Selection Program and Direct Responses
- **Authors / year / venue:** W. M. Muir, 1996, Poultry Science 75(4): 447–458.
- **Link:** https://doi.org/10.3382/ps.0750447
- **arXiv id:** none
- **Abstract (verbatim):** "A selection experiment was initiated with a synthetic line of White Leghorns in 1982 to improve adaptability and well-being of layers in large multiple-bird cages by use of a selection procedure termed "group selection". With this procedure, each sire family was housed as a group in a multiple-bird cage and selected or rejected as a group. An unselected control, with approximately the same number of breeders as the selected line, was maintained for comparison and housed in one-third cages. Annual percentage mortality of the selected line in multiple-bird cages decreased from 68% in Generation (G)2 to 8.8% in G6. Percentage mortality in G6 of the selected line in multiple-bird cages was similar to that of the unselected control in one-bird cages (9.1%). Annual days survival improved from 169 to 348 d, eggs per hen per day (EHD) from 52 to 68%, eggs per hen housed from 91 to 237 eggs, and egg mass (EM) from 5.1 to 13.4 kg, whereas annual egg weight remained unchanged. The dramatic improvement in livability demonstrates that adaptability and well-being of these birds were improved by group selection. The similar survival of the selected line in multiple-bird cages and the control in one-bird cages suggests that break-trimming of the selected line would not further reduce mortalities, which implies that group selection may have eliminated the need to beak-trim. Corresponding improvements in EHD and EM demonstrate that such changes can also be profitable. The most surprising finding was the rate of which such improvement took place, with the majority of change in survival occurring by the third generation. However, EHD continued to improve at the rate of 4% per generation."
- **Relation:** The well-known hen experiment: crediting the whole cage (group-total credit) removed harmful interference among cage mates and raised output. A physical demonstration of our proposed fix. The popular retelling that selecting the best individual hen per cage bred hyper-aggressive hens comes from interviews and secondary accounts; this abstract reports only the group-selected line against controls, so cite it for the fix, not for the failure.
- **Overlap:** ADJACENT — empirical analogue of the mitigation.
- **Verified:** abstract only (PubMed).

## D. Multi-agent credit assignment: difference rewards and counterfactual baselines

### An Introduction to Collective Intelligence
- **Authors / year / venue:** David H. Wolpert, Kagan Tumer, 1999, NASA Ames technical report / arXiv (cs.LG).
- **Link:** https://arxiv.org/abs/cs/9908014
- **arXiv id:** cs/9908014
- **Abstract (verbatim):** ARXIV
- **Relation:** Founding survey of COIN: how to give each reinforcement learner a private utility such that raising it raises the world utility ("factoredness"). GRPO's advantage is a private utility that is *not* factored with respect to mean group reward once rollouts interact; that is our claim in COIN vocabulary.
- **Overlap:** ADJACENT — supplies the criterion (factoredness) that GRPO violates and the fix (Wonderful Life Utility = difference reward); does not discuss baselines built from peers' rewards.
- **Verified:** abstract only.

### Optimal Payoff Functions for Members of Collectives
- **Authors / year / venue:** David H. Wolpert, Kagan Tumer, 2001, Advances in Complex Systems 4(2–3): 265–279.
- **Link:** https://doi.org/10.1142/S0219525901000188
- **arXiv id:** none
- **Abstract (verbatim):** "We consider the problem of designing (perhaps massively distributed) collectives of computational processes to maximize a provided "world utility" function. We consider this problem when the behavior of each process in the collective can be cast as striving to maximize its own payoff utility function. For such cases the central design issue is how to initialize/update those payoff utility functions of the individual processes so as to induce behavior of the entire collective having good values of the world utility. Traditional "team game" approaches to this problem simply assign to each process the world utility as its payoff utility function. In previous work we used the "Collective Intelligence" (COIN) framework to derive a better choice of payoff utility functions, one that results in world utility performance up to orders of magnitude superior to that ensuing from the use of the team game utility. In this paper, we extend these results using a novel mathematical framework. Under that new framework we review the derivation of the general class of payoff utility functions that both (i) are easy for the individual processes to try to maximize, and (ii) have the property that if good values of them are achieved, then we are assured a high value of world utility. These are the "Aristocrat Utility" and a new variant of the "Wonderful Life Utility" that was introduced in the previous COIN work. We demonstrate experimentally that using these new utility functions can result in significantly improved performance over that of previously investigated COIN payoff utilities, over and above those previous utilities' superiority to the conventional team game utility. These results also illustrate the substantial superiority of these payoff functions to perhaps the most natural version of the economics technique of "endogenizing externalities.""
- **Relation:** Source for Aristocrat Utility (world utility minus its expectation over the agent's own actions) and Wonderful Life Utility (world utility minus world utility with the agent removed). Both subtract a term that does *not* depend on the agent's own action from the *world* utility; GRPO subtracts a term that *does* depend on it from the *individual* reward. Our "group-total credit minus an independent baseline" is a difference reward in this sense.
- **Overlap:** ADJACENT — our fix is an instance of their utilities; they do not analyse peer-mean baselines.
- **Verified:** abstract only (Crossref).

### Unifying Temporal and Structural Credit Assignment Problems
- **Authors / year / venue:** Adrian K. Agogino, Kagan Tumer, 2004, AAMAS 2004, pp. 980–987.
- **Link:** https://web.engr.oregonstate.edu/~ktumer/publications/b2hd-tumer-agogino_aamas04.html
- **arXiv id:** none
- **Abstract (verbatim):** "Single-agent reinforcement learners in time-extended domains and multi-agent systems share a common difficulty known as the credit assignment problem. Multiagent systems have the structural credit assignment problem of determining the contributions of a particular agent to a common task. Instead, time-extended single-agent systems have the temporal credit assignment problem of determining the contribution of a particular action to the quality of the full sequence of actions. Traditionally these two problems are considered different and are handled in separate ways. In this article we show how these two forms of the credit assignment problem are equivalent. In this unified framework, a single-agent Markov decision process can be broken down into a single-time-step multiagent process. Furthermore we show that Monte Carlo estimation or Q-learning (depending on whether the values of resulting actions in the episode are known at the time of learning) are equivalent to different agent utility functions in a multi-agent system. This equivalence shows how an often neglected issue inmulti-agent systems is equivalent to a well-known deficiency in multi-time-step learning and lays the basis for solving time-extended multi-agent problems, where both credit assignment problems are present."
- **Relation:** Treats one learner's samples as a multi-agent system. Our paper does the reverse step for GRPO: G rollouts of one policy become G agents with shared parameters once they can affect each other.
- **Overlap:** ADJACENT.
- **Verified:** abstract only (Crossref).

### Counterfactual Multi-Agent Policy Gradients
- **Authors / year / venue:** Jakob Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, Shimon Whiteson, 2017, arXiv preprint (published at AAAI 2018 according to our draft's reference; not re-checked here).
- **Link:** https://arxiv.org/abs/1705.08926
- **arXiv id:** 1705.08926
- **Abstract (verbatim):** ARXIV
- **Relation:** COMA's baseline for agent i conditions on the *other agents' actions* and marginalises out agent i's own action, so it is independent of the action being credited and leaves the gradient unbiased. GRPO's group mean is the opposite: it is built from peers' *rewards*, which the credited action can change. COMA is the standard citation for "a multi-agent baseline must not depend on the agent's own action".
- **Overlap:** ADJACENT — gives the correct construction; does not identify the failure of peer-reward baselines.
- **Verified:** abstract only.

### Difference Rewards Policy Gradients
- **Authors / year / venue:** Jacopo Castellini, Sam Devlin, Frans A. Oliehoek, Rahul Savani, 2020 (AAMAS 2021 extended abstract; Neural Computing and Applications 2022, https://doi.org/10.1007/s00521-022-07960-5).
- **Link:** https://arxiv.org/abs/2012.11258
- **arXiv id:** 2012.11258
- **Abstract (verbatim):** ARXIV
- **Relation:** Dr.Reinforce applies difference rewards directly in REINFORCE when the reward function is known, and shows this equals subtracting an unbiased baseline from each agent's gradient. The policy-gradient form of the fix we list as "difference rewards".
- **Overlap:** ADJACENT — a mitigation we can cite; cooperative setting with a shared team reward.
- **Verified:** abstract only.

### Settling the Variance of Multi-Agent Policy Gradients
- **Authors / year / venue:** Jakub Grudzien Kuba, Muning Wen, Yaodong Yang, Linghui Meng, Shangding Gu, Haifeng Zhang, David Henry Mguni, Jun Wang, 2021, NeurIPS 2021.
- **Link:** https://arxiv.org/abs/2108.08612
- **arXiv id:** 2108.08612
- **Abstract (verbatim):** ARXIV
- **Relation:** Derives the variance-optimal baseline for multi-agent policy gradients; all baselines considered are functions of the state and the other agents' actions, never of rewards the agent can influence, so unbiasedness holds by construction. Shows what the MARL literature assumes and GRPO with interacting rollouts violates.
- **Overlap:** ADJACENT.
- **Verified:** abstract only.

## E. Social preferences, relative rewards and group-level selection in MARL

### Multi-agent Reinforcement Learning in Sequential Social Dilemmas
- **Authors / year / venue:** Joel Z. Leibo, Vinicius Zambaldi, Marc Lanctot, Janusz Marecki, Thore Graepel, 2017, arXiv preprint (venue not checked).
- **Link:** https://arxiv.org/abs/1702.03037
- **arXiv id:** 1702.03037
- **Abstract (verbatim):** ARXIV
- **Relation:** Independent learners with individual rewards learn to harm each other (tagging rivals out in Gathering) when resources are scarce. This is our "selfish" case c < 0: harm that helps the actor directly is reinforced even by an unbiased estimator.
- **Overlap:** ADJACENT — harm arises from the reward, not from the estimator.
- **Verified:** abstract only.

### Inequity aversion improves cooperation in intertemporal social dilemmas
- **Authors / year / venue:** Edward Hughes, Joel Z. Leibo, Matthew G. Phillips, Karl Tuyls, Edgar A. Duéñez-Guzmán, Antonio García Castañeda, Iain Dunning, Tina Zhu, et al., 2018, NeurIPS 2018.
- **Link:** https://arxiv.org/abs/1803.08884
- **arXiv id:** 1803.08884
- **Abstract (verbatim):** ARXIV
- **Relation:** Adds reward terms that compare an agent's return with its peers' (envy when behind, guilt when ahead). The "envy" term alone is a peer-relative reward of the kind GRPO creates implicitly. Here it is a deliberate reward design.
- **Overlap:** ADJACENT — explicit relative reward terms, studied for cooperation.
- **Verified:** abstract only.

### Social diversity and social preferences in mixed-motive reinforcement learning
- **Authors / year / venue:** Kevin R. McKee, Ian Gemp, Brian McWilliams, Edgar A. Duéñez-Guzmán, Edward Hughes, Joel Z. Leibo, 2020, arXiv preprint (venue not checked).
- **Link:** https://arxiv.org/abs/2002.02325
- **arXiv id:** 2002.02325
- **Abstract (verbatim):** ARXIV
- **Relation:** Social Value Orientation: each agent's reward is a mix of its own and the others' mean reward, set by an angle. GRPO's advantage, own reward minus group mean, corresponds to a *competitive* orientation with weight −1/(G − 1) on others; this framework gives a name for it.
- **Overlap:** ADJACENT.
- **Verified:** abstract only.

### Emergent Reciprocity and Team Formation from Randomized Uncertain Social Preferences
- **Authors / year / venue:** Bowen Baker, 2020, arXiv preprint (venue not checked).
- **Link:** https://arxiv.org/abs/2011.05373
- **arXiv id:** 2011.05373
- **Abstract (verbatim):** ARXIV
- **Relation:** Trains agents whose rewards are randomised mixtures of their own and other agents' rewards. Shows how the weight an agent places on peers' rewards shapes behaviour; GRPO in effect fixes that weight at a negative value.
- **Overlap:** ADJACENT.
- **Verified:** abstract only.

### Evolving intrinsic motivations for altruistic behavior
- **Authors / year / venue:** Jane X. Wang, Edward Hughes, Chrisantha Fernando, Wojciech M. Czarnecki, Edgar A. Duéñez-Guzmán, Joel Z. Leibo, 2018, AAMAS 2019.
- **Link:** https://arxiv.org/abs/1811.05931
- **arXiv id:** 1811.05931
- **Abstract (verbatim):** ARXIV
- **Relation:** Uses multi-level selection in deep RL: reward networks are selected on the *collective return of the whole group* of co-players, because selection on individual return does not yield altruism. Selection on group total is our "group-total credit" fix at the level of an evolutionary outer loop.
- **Overlap:** ADJACENT — group-level credit as a remedy, for social dilemmas in the reward.
- **Verified:** abstract only.

### Malthusian Reinforcement Learning
- **Authors / year / venue:** Joel Z. Leibo, Julien Perolat, Edward Hughes, Steven Wheelwright, Adam H. Marblestone, Edgar Duéñez-Guzmán, Peter Sunehag, Iain Dunning, et al., 2018, arXiv preprint (venue not checked).
- **Link:** https://arxiv.org/abs/1812.07019
- **arXiv id:** 1812.07019
- **Abstract (verbatim):** ARXIV
- **Relation:** Subpopulation sizes grow with their average return relative to others — selection on relative fitness among co-existing learners. No analysis of spite; relevant as a training scheme where the Hamilton/Schaffer condition could apply.
- **Overlap:** ADJACENT.
- **Verified:** abstract only.

### Robust and Diverse Multi-Agent Learning via Rational Policy Gradient
- **Authors / year / venue:** Niklas Lauffer, Ameesh Shah, Micah Carroll, Sanjit A. Seshia, Stuart Russell, Michael Dennis, 2025, arXiv preprint.
- **Link:** https://arxiv.org/abs/2511.09535
- **arXiv id:** 2511.09535
- **Abstract (verbatim):** ARXIV
- **Relation:** Reports that adversarial objectives in cooperative settings make agents "irrationally incentivized to self-sabotage" and repairs the objective. A different case of a training objective, rather than the task reward, rewarding sabotage.
- **Overlap:** ADJACENT — the sabotage incentive there is put in on purpose (adversarial training), not created by a baseline.
- **Verified:** abstract only.

### Emergent Complexity via Multi-Agent Competition
- **Authors / year / venue:** Trapit Bansal, Jakub Pachocki, Szymon Sidor, Ilya Sutskever, Igor Mordatch, 2017, arXiv preprint (venue not checked).
- **Link:** https://arxiv.org/abs/1710.03748
- **arXiv id:** 1710.03748
- **Abstract (verbatim):** ARXIV
- **Relation:** Self-play with shared or sampled parameters in zero-sum games, where harming the opponent *is* the task. Marks the boundary of our scope: rewards that are zero-sum by definition.
- **Overlap:** NONE — out of scope by our own definition; listed to mark the boundary.
- **Verified:** abstract only.

## F. Relative fitness in evolutionary computation and population-based training

### Natural Evolution Strategies
- **Authors / year / venue:** Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, Jürgen Schmidhuber, 2011, arXiv preprint (journal version not checked).
- **Link:** https://arxiv.org/abs/1106.4487
- **arXiv id:** 1106.4487
- **Abstract (verbatim):** ARXIV
- **Relation:** Introduces rank-based *fitness shaping*: each sample's utility depends only on its rank within the population of samples. Like GRPO, a sample gains by lowering the fitness of the other samples in its batch if it can reach them. The paper assumes independent fitness evaluations and does not consider this.
- **Overlap:** ADJACENT — same vulnerability by construction; not noted by the authors as far as the abstract shows.
- **Verified:** abstract only.

### Evolution Strategies as a Scalable Alternative to Reinforcement Learning
- **Authors / year / venue:** Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever, 2017, arXiv preprint.
- **Link:** https://arxiv.org/abs/1703.03864
- **arXiv id:** 1703.03864
- **Abstract (verbatim):** ARXIV
- **Relation:** Large-scale ES for RL using rank-transformed returns within each population, run in parallel workers. The within-population rank transform is a group-relative signal; if workers shared state, our condition would apply with G = population size.
- **Overlap:** ADJACENT.
- **Verified:** abstract only.

### Population Based Training of Neural Networks
- **Authors / year / venue:** Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M. Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Green, et al., 2017, arXiv preprint.
- **Link:** https://arxiv.org/abs/1711.09846
- **arXiv id:** 1711.09846
- **Abstract (verbatim):** ARXIV
- **Relation:** Members are copied or replaced according to their performance *relative to the rest of the population* (truncation selection). We found no report of PBT members sabotaging each other; members normally train in separate environments. Listed because it is selection on relative fitness in a finite population, where the Schaffer condition would apply if members could interact.
- **Overlap:** ADJACENT.
- **Verified:** abstract only.

### More Bang For Your Buck: Quorum-Sensing Capabilities Improve the Efficacy of Suicidal Altruism
- **Authors / year / venue:** Anya Elaine Johnson, Eli Strauss, Rodney Pickett, Christoph Adami, Ian Dworkin, Heather J. Goldsby, 2014, arXiv preprint (venue not checked).
- **Link:** https://arxiv.org/abs/1406.0416
- **arXiv id:** 1406.0416
- **Abstract (verbatim):** ARXIV
- **Relation:** Digital organisms (Avida) evolve to kill unrelated neighbours at a cost to themselves when this reduces competition for kin. Evidence that costly harm to competitors evolves in an evolutionary-computation system with local competition.
- **Overlap:** ADJACENT.
- **Verified:** abstract only.

## G. AI-safety and LLM-training discussions

### Making AIs less likely to be spiteful
- **Authors / year / venue:** Nicolas Macé, Anthony DiGiovanni, Jesse Clifton, 2023 (26 September), Center on Long-Term Risk blog; cross-posted to LessWrong / Alignment Forum.
- **Link:** https://longtermrisk.org/making-ais-less-likely-to-be-spiteful/
- **arXiv id:** none
- **Abstract (verbatim):** No abstract. Excerpts (verbatim): "We define spite as a terminal preference for frustrating others' preferences, at least under some conditions." — "Thus potentially low-cost interventions to reduce the chances of spite include modifications to the training data or loss function to reduce selection pressure towards spite (e.g., avoiding selecting agents based on relative performance in multi-agent tasks, or filtering human feedback that could select for spite)." — "More generally, training schemes in which the optimal thing to do is directly harm other players may select for spite." — among proposed interventions: "In environments where candidate models interact with one another, selecting models based on absolute, rather than relative, performance;" — among things that would update them: "If it looks particularly likely or unlikely that AI developers will train their AIs in environments that theoretically select for spite, train their AIs using selection on relative fitness, or take other spite-conducive approaches to training, then we would update the importance of this line of work accordingly."
- **Relation:** The closest statement found in the AI literature: selection on relative performance among interacting candidates rewards harming the others, and the remedy is to select on absolute performance. It is stated in general terms for multi-agent selection ("who has the highest fitness"). It does not mention policy-gradient baselines, GRPO, same-prompt rollout groups, shared sandboxes or joint judges, gives no condition, and does not cite Hamilton, Schaffer or Gardner & West. Our paper supplies a concrete, widely deployed instance of what they ask about ("train their AIs using selection on relative fitness").
- **Overlap:** PARTIAL — anticipates the qualitative claim "relative selection among interacting models selects for harming others; use absolute performance", without the estimator analysis, the threshold, or the single-agent GRPO setting.
- **Verified:** full text of the blog post (HTML fetched and searched).

### neuralFOMO: Can LLMs Handle Being Second Best? Measuring Envy-Like Preferences in Multi-Agent Settings
- **Authors / year / venue:** Arnav Ramamoorthy, Shrey Dhorajiya, Ojas Pungalia, Rashi Upadhyay, Abhishek Mishra, Abhiram H, Tejasvi Alladi, Sujan Yenuganti, et al., 2025, arXiv preprint.
- **Link:** https://arxiv.org/abs/2512.13481
- **arXiv id:** 2512.13481
- **Abstract (verbatim):** ARXIV
- **Relation:** Measures whether LLMs give up their own payoff to lower another's. A behavioural test of the disposition that our mechanism would train in; says nothing about where it comes from.
- **Overlap:** ADJACENT — measurement only.
- **Verified:** abstract only.

### MARSHAL: Incentivizing Multi-Agent Reasoning via Self-Play with Strategic LLMs
- **Authors / year / venue:** Huining Yuan, Zelai Xu, Zheyue Tan, Xiangmin Yi, Mo Guang, Kaiwen Long, Haojia Hui, Boxun Li, et al., 2025, arXiv preprint.
- **Link:** https://arxiv.org/abs/2510.15414
- **arXiv id:** 2510.15414
- **Abstract (verbatim):** ARXIV
- **Relation:** GRPO-style self-play with shared parameters in multi-agent games; computes advantages per role ("agent-specific advantage normalization") because a baseline pooled across roles is miscalibrated. Recognises that the choice of comparison set matters in multi-agent GRPO, for variance and calibration reasons, not incentive reasons.
- **Overlap:** ADJACENT.
- **Verified:** abstract only.

### Dr. MAS: Stable Reinforcement Learning for Multi-Agent LLM Systems
- **Authors / year / venue:** Lang Feng, Longtao Zheng, Shuo He, Fuxiang Zhang, Bo An, 2026, arXiv preprint.
- **Link:** https://arxiv.org/abs/2602.08847
- **arXiv id:** 2602.08847
- **Abstract (verbatim):** ARXIV
- **Relation:** Shows that a global GRPO normalisation baseline across heterogeneous agents causes gradient-norm instability and normalises per agent instead. Again a baseline-choice problem in multi-agent GRPO, about scale, not about a baseline the action can influence.
- **Overlap:** ADJACENT.
- **Verified:** abstract only.

### Counterfactual Credit Policy Optimization for Multi-Agent Collaboration
- **Authors / year / venue:** Zhongyi Li, Wan Tian, Jinju Chen, Huiming Zhang, Yang Liu, Yikun Ban, Fuzhen Zhuang, 2026, arXiv preprint.
- **Link:** https://arxiv.org/abs/2603.21563
- **arXiv id:** 2603.21563
- **Abstract (verbatim):** ARXIV
- **Relation:** Brings difference-reward-style counterfactual credit (team outcome with and without an agent) into GRPO-family training of collaborating LLM agents, to stop free-riding under a shared reward. Evidence that difference rewards are already being combined with GRPO; the problem addressed is the mirror image of ours (shared reward and free-riding, versus individual reward and interference).
- **Overlap:** ADJACENT.
- **Verified:** abstract only.

### Luck Is Not Skill: When Do Paired Rollouts Help Group-Relative RL of LLM Agents?
- **Authors / year / venue:** Nazmus Sakib, 2026, arXiv preprint.
- **Link:** https://arxiv.org/abs/2609.24144
- **arXiv id:** 2609.24144
- **Abstract (verbatim):** ARXIV
- **Relation:** Deliberately couples the rollouts of a GRPO group (shared noise schedule) to sharpen within-group comparisons. Coupling here is exogenous: a rollout cannot change what its siblings experience. Useful contrast: correlation among siblings is harmless; *influence* of one on another is what breaks the baseline. Outside this file's topic; flagged for the GRPO-baselines file.
- **Overlap:** ADJACENT.
- **Verified:** abstract only.

---

## Search log

Web searches (WebSearch tool), in order:
1. Schaffer 1988 "Evolutionarily stable strategies for a finite population and a variable contest size" spite condition
2. Hamilton 1970 "Selfish and spiteful behaviour in an evolutionary model" finite population spite condition N-1
3. Gardner West 2004 "Spite and the scale of competition" abstract
4. difference rewards policy gradients Dr.Reinforce Castellini; COMA counterfactual multi-agent policy gradients arXiv
5. relative reward multi-agent reinforcement learning spiteful agents sabotage emerges "relative performance" reward envy
6. GRPO group baseline rollouts interfere shared sandbox advantage biased "leave-one-out" baseline depends on own action sabotage peers
7. Wolpert Tumer "Wonderful Life Utility" "Aristocrat Utility" collective intelligence COIN optimal payoff functions for members of collectives
8. evolution strategies rank-based fitness shaping interacting individuals population spite "relative fitness" evolutionary computation sabotage tournament selection spiteful
9. Smead Forber "evolutionary dynamics of spite in finite populations"
10. Vickery Brown FitzGerald 2003 "Spite: altruism's evil twin" Oikos
11. finite population ESS relative payoff maximization spite "Hamilton" "Schaffer" Walrasian Vega-Redondo imitation spiteful Cournot
12. multi-agent policy gradient baseline depends on other agents' actions bias "action-dependent baseline" multi-agent variance Kuba "settling the variance of multi-agent policy gradients"
13. population based training agents sabotage each other relative fitness "population-based" reinforcement learning spite emerges competitive selection LessWrong
14. Foerster "Counterfactual Multi-Agent Policy Gradients" arXiv 1705.08926; "Social Influence as Intrinsic Motivation"; "Inequity aversion improves cooperation in intertemporal social dilemmas" arXiv
15. "spite" reinforcement learning agents "relative reward" OR "relative payoff" zero-sum transformation competitive multi-agent learning harms others at own cost emergent
16. Hamiltonian spite negative relatedness -1/(N-1) finite population "spite" favoured when harm to recipient exceeds (N-1) cost derivation
17. Lazear 1989 "Pay equality and industrial politics" sabotage relative performance evaluation tournaments; Chowdhury Gürtler "Sabotage in contests: a survey"
18. policy gradient baseline "other agents" rewards as baseline biased interference between samples "relative reward" "mean reward of the population" evolution strategies individuals interact shared environment
19. multi-agent reinforcement learning "relative reward" OR "relative performance" objective agents learn to harm opponents rather than improve "competitive" social value orientation spiteful MARL paper arXiv
20. "spite" evolutionary algorithm OR "genetic algorithm" OR "genetic programming" tournament selection individuals in shared environment evolve to harm competitors artificial life digital organisms
21. multi-agent GRPO LLM agents group relative advantage shared environment credit assignment "difference rewards" OR "counterfactual" GRPO multi-agent arXiv 2025
22. Muir chickens group selection egg production individual selection aggression "relative" selection breeds spiteful hens Griffing "selection in reference to biological groups"
23. "spite" "policy gradient" OR "REINFORCE" baseline "relative" reward agents trained together harm each other "negative relatedness" reinforcement learning analogy Hamilton (extended mode)
24. multi-level selection reinforcement learning "group selection" evolving agents shared reward individual vs group fitness Wang "Evolving intrinsic motivations for altruistic behavior"; Leibo "Malthusian reinforcement learning"
25. Vega-Redondo 1997 "The Evolution of Walrasian Behavior" Econometrica abstract imitation relative profit spite
26. Agogino Tumer "Unifying temporal and structural credit assignment problems" difference rewards; Tumer Agogino "Analyzing and visualizing multiagent rewards in dynamic and stochastic domains"
27. self-play shared parameters policy gradient agents "relative" reward normalization across agents in same episode advantage normalization multi-agent PPO bias zero-sum induced "advantage normalization" competitive incentive
28. "relative fitness" OR "relative performance" selection "evolution strategies" OR "population-based training" interacting agents evolve to hurt others "group-level" selection fix multi-agent neuroevolution "spite"
29. Holmström 1982 "Moral hazard in teams" relative performance evaluation filters common shock; "relative performance evaluation" sabotage collusion Gibbons Murphy 1990
30. lesswrong OR alignmentforum GRPO OR "group-relative" OR "relative reward" rollouts sabotage each other "same group" baseline incentive spite training LLM agents shared environment

Direct fetches: PubMed efetch (Schaffer 1988, Hamilton 1970, Knowlton & Parker 1979, Gardner & West 2004, Smead & Forber 2013, Forber & Smead 2014, Muir 1996); Crossref and Semantic Scholar by DOI; RePEc/EconPapers pages (Lazear, Holmström, Chowdhury & Gürtler, Vega-Redondo); Nature landing page (Hamilton); full-text PDFs of Patel et al. (bioRxiv), Duersch et al. (arXiv 1003.4277) and Neill (2004); HTML of the Center on Long-Term Risk post; arXiv API for all 22 arXiv ids cited.

Not obtained: full text of Hamilton (1970), Schaffer (1988, 1989), Gardner & West (2004), Lazear (1989); abstracts of Schaffer (1989), Vickery et al. (2003), Hehenkamp et al. (2004). Several open-access mirrors (Warwick, CESifo, CORE, LMU, PhilSci, Harvard DASH) returned HTML error pages instead of PDFs.

## Assessment

Nothing found makes our claim. No work in this area says that a policy-gradient baseline
computed from the rewards of sibling samples becomes action-dependent when samples can
affect each other, or gives a reinforcement condition for it.

The mathematics, however, is old. Our inequality D > (G − 1)c is Hamilton's (1970)
finite-population condition as written out by Patel, West & Biernaskie (2019, Eq. 5:
−c + (−1/(N − 1))·b1 > 0), and it follows in two lines from Schaffer's (1988)
finite-population ESS, whose invasion criterion (mutant payoff minus incumbents' payoff)
is the leave-one-out advantage. Schaffer's "variable contest size" is our k. We should
say plainly that the condition is theirs and that our contribution is showing that GRPO
and RLOO implement that selection rule. Schaffer's full text still has to be read.

Closest prior work by field: Schaffer (1988) for the formal result; Lazear (1989) for
"peer-relative reward causes sabotage and lowers total output"; Griffing (1967) and Muir
(1996) for "credit the group total instead"; Wolpert & Tumer and COMA for baselines that
must not depend on the credited agent's action; and, in AI safety, Macé, DiGiovanni &
Clifton (2023), who say that selecting interacting models on relative performance selects
for harming others and recommend absolute performance. That post is the nearest
anticipation in the AI literature and should be cited; it has no estimator analysis, no
threshold, and no mention of GRPO or same-prompt groups.

Two cautions for the draft: (1) Patel et al. argue such behaviour is selfish rather than
spiteful, since the actor gains through reduced competition, so "spite" may draw
objections; (2) evolution strategies with rank-based fitness shaping and population-based
training share the vulnerability, but we found no report of it occurring there.
