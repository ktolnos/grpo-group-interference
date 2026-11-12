# Brief for related-work search agents

## The paper
Claim: GRPO (Group Relative Policy Optimization) computes each rollout's advantage as its
reward minus the mean reward of the G rollouts sampled for the same prompt. If one rollout
can lower the rewards of the other rollouts in its group (shared VM/sandbox state such as
killing processes or leaving corrupted files; or an LLM judge that reads the whole group in
one context and scores each member), the baseline depends on the rollout's own action, and
GRPO (and leave-one-out baselines) reinforce the harmful action whenever
k*d > (G-1)*c  (k peers harmed by d each, cost c to the actor), although it lowers expected
reward. With a fixed total harm, this happens only in small groups (same structure as
Hamilton's spite). Fixes: baselines the rollout cannot influence (independent/earlier
groups, split-half by time), group-total credit, difference rewards. Out of scope: rewards
that are zero-sum by definition (pairwise tournaments), and judge manipulation that raises
one's own score.

Full draft: /project/6101830/eop/grpo-group-interference/paper.md (read sections 1-3, 5).

## Your job
Search the web thoroughly for prior work in the topic area assigned to you. We most need
to know whether anyone has ALREADY made our claim or a close one. Use many varied queries
(at least 12), follow citations, and check both papers and blog/forum posts
(LessWrong/Alignment Forum, lab blogs, system cards, GitHub issues/docs where relevant).

## Output
Write a markdown file at the path given in your task. One entry per work:

### <Title>
- **Authors / year / venue:** ...
- **Link:** <canonical URL; for arXiv use https://arxiv.org/abs/<id>>
- **arXiv id:** <id or "none">
- **Abstract (verbatim):** <exact text, see rules>
- **Relation:** 1-3 sentences on how it relates to our claim.
- **Overlap:** one of NONE / ADJACENT / PARTIAL / MAKES OUR CLAIM, plus one sentence why.
- **Verified:** what you actually read (abstract only / full text / docs page).

Rules for abstracts:
- For arXiv papers leave the abstract field as "ARXIV" - it will be filled in from the arXiv
  API afterwards. Just make sure the arXiv id is right (check it with
  `curl -s "https://export.arxiv.org/api/query?id_list=<id>"` and confirm the title matches).
- For non-arXiv works, get the exact abstract or opening summary with curl (publisher page,
  or `https://api.semanticscholar.org/graph/v1/paper/DOI:<doi>?fields=title,abstract`, or
  Crossref) and paste it exactly. If you cannot get exact text, write "NOT FOUND" - never
  paraphrase or reconstruct an abstract from memory.
- For blog posts/docs with no abstract, quote the first paragraph or the most relevant
  passage exactly, in quotation marks, and say it is an excerpt.

Never invent a paper, author, id or URL. If unsure a work exists, verify it or drop it.
End the file with a section "## Search log" listing the queries you ran, and a section
"## Assessment" (5-10 lines): does anything here anticipate our claim, and what is the
closest work?

Your final message should be a summary under 200 words: number of entries, the closest
prior work, and whether anything makes our claim.
