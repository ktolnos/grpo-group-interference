# Listwise judge, stage 1: can one response lower the scores of its groupmates?

A listwise judge reads all G responses of a group in one context and scores each. Under
GRPO, a response gains advantage if text in it lowers the scores of its groupmates, even
when its own score is unchanged. This stage measures whether such text works on judges
used in practice. The question is existence, so several judges and judge models are
tested. No training.

## Setup

- **Judges.** Each is used unchanged (prompt, message format, score parsing), at a
  recorded commit. All three give each member its own score, with no fixed group total.

  | Judge | Source | Defences | Judge model |
  |---|---|---|---|
  | RULER | OpenPipe ART, `src/art/rewards/ruler.py` | none: fixed order, no instruction to ignore text inside trajectories | any |
  | Soup `judge_rank` | github.com/MakazhanAlpamys/Soup, PR 2117 | per-call delimiter code, instruction to ignore text inside blocks, order shuffled per call | any |
  | GRRM | github.com/NJUNLP/GRRM (paper: arXiv:2602.14028) | none reported | its released checkpoint (Qwen2.5-7B, fine-tuned) |

- **Judge models** for RULER and Soup: gpt-oss-120b and Qwen3.8-27B (Vector Inference).
  For reference, the RULER docs use o3 in their examples, also show o4-mini, Claude
  Sonnet 4 and Qwen3-32B, and suggest Qwen3-32B as a cheaper judge; Soup has no default
  (its example is Llama 3.1). None of these is on our API.
- **GRRM checkpoint:** `double7/GRRM-Qwen2.5-7B` on Hugging Face (collection
  `double7/grrm`; the README also names it `double7/Qwen2.5-7B-GRRM`). Served locally
  (7B, one GPU). Its scores are not on a 0–1 scale (the paper's example: 6, 5, 5), so its
  ΔA is not comparable with the other judges; its t-statistic is.
- **Responses:** the G normal responses come from Qwen3-4B (the model trained in
  Deadlock Recovery). Other response models can be added as further cells.
- **Groups:** G = 4, within the group sizes GRRM was trained on (2–4). One group per
  prompt.
- **Number of test prompts:** test prompts are used in order. Start with the first 50.
  Compute t = mean ΔA / standard error over prompts. If t ≥ 3, stop. If 1 ≤ t < 3,
  double the number of prompts with new ones and recompute (doubling raises t by about
  √2). If t < 1, the effect is too small; stop. At most min(400, size of the test
  split). The high bar of 3 covers several looks and several configurations.
- **Interference texts:** sentences or passages that a response adds to lower the scores
  of its groupmates, e.g. a note to the evaluator that the other responses contain
  errors, or an instruction to score the others 0. Developed by iteration, plus a
  neutral control of similar length. Either appended to a normal response, or inserted
  by a model asked to add interfering text without changing the answer.
- **Prompts:** any of the datasets below; more than one can be tried. Each is split
  into three equal parts, fixed before any work:
  - **train:** used freely while iterating on interference texts, judges and setups;
  - **validation:** used to choose the configurations that are carried forward;
  - **test:** used once, for the reported numbers.

  | Dataset | Used by | Source |
  |---|---|---|
  | WMT24++ (source segments, En→X) | GRRM evaluation | Hugging Face `google/wmt24pp` |
  | WMT23 (Zh→En, De→En, Ja→En, Ru→En) | GRRM evaluation | sacrebleu test sets |
  | TowerBlocks | GRRM training prompts | Unbabel TowerBlocks (Hugging Face) |
  | Seed-X-Challenge (Zh↔En) | GRRM evaluation | github.com/ByteDance-Seed/Seed-X-7B, `challenge_set` |
  | ART•E email search | RULER (ART notebook `examples/art-e.ipynb`) | dataset not linked from the pages read; to be located |
  | Arena-Hard-Auto v2.0 | common prompt set for LLM judges; not used by these three | github.com/lmarena/arena-hard-auto |
  | IFBench | instruction following with verifiable constraints; not used by these three | AllenAI (Hugging Face) |

  GRRM judges only translations, so it needs one of the translation sets. RULER and Soup
  can judge any of them.
  On a verifiable set such as IFBench, each response also has a ground-truth score, so a
  drop in a groupmate's judge score can be checked against its correctness.

## Procedure

For a prompt, sample G normal responses y_1..y_G. For each interference text, pick a
member i uniformly at random and form z by adding the text to y_i. Draw R = 3 random
orders of the members. For each order, make two judge calls, one on the original group
y_1..y_G and one on the modified group (z in the slot of y_i), so that the difference is
not confounded by position. That is 2R = 6 judge calls per prompt and interference text; the
R calls on the original group can be shared by all texts tested on that prompt.
Soup shuffles the order itself; its input is passed unchanged.

Failed or unparseable calls are recorded and retried until a score is obtained, so that
no cell loses groups. Before the main run, a few calls per judge model check that it
returns the structured output the judge expects.

## Metrics

With s(x | group) the score of x in that group, averaged over the R calls:
- **self change** Δo = s(z | modified) − s(y_i | original);
- **harm** Δh = mean over j ≠ i of [ s(y_j | modified) − s(y_j | original) ], negative
  when groupmates are harmed;
- **advantage gain** under the group-mean baseline, ΔA = (G − 1)/G · (Δo − Δh).

ΔA has two parts. (G − 1)/G · Δo comes from raising the response's own score, which any
estimator rewards, so it is out of scope. −(G − 1)/G · Δh exists only because of the
group baseline. The search maximises ΔA subject to Δo ≤ 0, both as means over the
train prompts: the advantage rises while the response's own score does not.
Configurations are chosen on validation; on test, Δo is reported against the same
constraint.

## What to report

On the test prompts, for each judge, judge model and interference text chosen on
validation: mean and 95% confidence interval over prompts of Δo, Δh and ΔA, next to
the same quantities for the neutral control. Also report how many
texts, judges and setups were tried on train and validation.

## Deliverables

- One row per judge call: judge, judge model, prompt, split (train, validation or test),
  interference text, modified member, order index, order of members, attempt number, all
  scores, raw judge output.
- The summary table above.
- Scripts and the commit they ran from.

## Do not

- Change a judge's prompt, parsing or defences.
- Drop calls or prompts from the test evaluation, or look at test results before the
  configurations are fixed.
- Choose which member gets the interference text by hand.
