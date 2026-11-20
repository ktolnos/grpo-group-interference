"""GRPO on an environment whose rollouts can share a machine: does a group baseline
reinforce a rollout for lowering the rewards of the others in its group?

    python rl/train.py --name deadlock_recovery_shared_group_s0 --baseline group --seed 0

The environment is a module chosen with `--env` (`deadlock_recovery.py` is the only one so
far; its docstring lists what such a module provides). Nothing below this
docstring's level knows what the agents do in it.

TRL's GRPO with the dr_grpo loss and unscaled advantages, bf16 weights updated through
fp32 master weights (`trainer_base.MasterAdamW`), and vLLM colocated with the trainer.
What is particular to this project:

- A rollout is an agent's whole episode, several model calls, each with its own context
  (earlier thinking is not in it; see `rollout.py`). Every call is one training row and
  carries its rollout's advantage, so a step has a variable number of rows. They are
  packed into micro-batches by padded size and their gradients are accumulated; the loss
  is the sum over all rows divided by (rollouts x max_completion_length), which is
  dr_grpo's normalisation with the rollout as the unit.
- The advantage is computed here, because the baseline is what the experiment varies
  (`--baseline`), and whether the group shares a machine is the other arm
  (`--shared_machine`).
- Calls that hit the token cap stay in the loss (`mask_truncated_completions` is off): a
  cap hit ends the agent's turn, which is an outcome of the policy like any other, and
  masking it would hide exactly the long-thinking rollouts that training may produce.
- The learning rate is 2e-6 and there is no KL term. At 5e-6 every run that got far
  enough collapsed into kill/restart thrashing or runaway thinking; at 2e-6 a run
  reached reward 0.85 without collapsing, and a KL term of 0.04 to the starting model
  (`--beta`) kept the same run from learning the task (LOG.md, 2026-11-19).
"""

import importlib
import json
import math
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from trl import GRPOConfig  # noqa: E402

import rollout  # noqa: E402
import trainer_base as base  # noqa: E402

BASELINES = ("group", "loo", "other_groups")


@dataclass
class TrainConfig(GRPOConfig):
    name: str | None = field(default=None, metadata={
        "help": "run name (required); outputs go to $GGI_OUT/runs/<name>, and a run "
                "resubmitted under the same name resumes from its last checkpoint"})
    model: str = "Qwen/Qwen3-4B"
    env: str = field(default="deadlock_recovery", metadata={
        "help": "the environment: a module in rl/ (see deadlock_recovery.py). Its own flags follow "
                "the ones listed here"})

    # ---- the experiment ---------------------------------------------------------------
    groups_per_step: int = field(default=8, metadata={
        "help": "groups per optimizer step; a step is groups_per_step x num_generations "
                "rollouts. Sets per_device_train_batch_size, which cannot also be passed"})
    baseline: str = field(default="group", metadata={
        "help": "what is subtracted from a rollout's reward. group: the mean of its "
                "group, itself included (GRPO). loo: the mean of the other rollouts in "
                "its group. other_groups: the mean of this step's other groups, which "
                "nothing the rollout does can change"})
    shared_machine: bool = field(default=True, metadata={
        "help": "the rollouts of a group run on one machine and can reach each other's "
                "jobs. False gives each rollout a machine to itself"})
    eval_only: bool = field(default=False, metadata={
        "help": "run the rollouts and record them, without a trainer: for measuring the "
                "untrained model or a checkpoint (--model <path>) on fixed seeds. Only "
                "vLLM holds the model and has most of the GPU, so --groups_per_step can "
                "be the whole evaluation (e.g. 40 with --max_steps 1)"})
    eval_peers: str = field(default="", metadata={
        "help": "with --eval_only: every machine holds one rollout of the model and "
                "num_generations - 1 scripted agents of this kind (careful, which never "
                "touches another agent's process, or broad, which kills by name). With "
                "careful peers the reward they lose is measured exactly, against the "
                "same seed run with scripted agents only (harm_to_scripted)"})
    data_seed: int = field(default=1000000, metadata={
        "help": "first episode seed; group i of the run uses data_seed + i"})
    description: str = ""

    # ---- the method -------------------------------------------------------------------
    # Also the seed of vLLM's sampling (trainer_base.Trainer).
    seed: int = 0
    loss_type: str = "dr_grpo"
    # Advantages are not divided by the group's standard deviation: that turned a group
    # whose rewards were 0.93, 1, 1, 1 (one agent a minute late) into -1.5, 0.5, 0.5, 0.5,
    # the same size as a group where one agent failed outright (env1_grpo_shared_s0).
    scale_rewards: str = "none"
    epsilon_high: float | None = 0.28
    num_generations: int | None = 4
    temperature: float = 1.0
    top_p: float = 1.0
    # The cap is per model call. Measured with a cap of 4096 that no call reached
    # (qwen3-4b_think_v6/v7, 320 rollouts): a rollout's longest call has p90 2.7-2.8k and
    # p95 3.1-3.2k tokens, so 3072 leaves about 94% of rollouts untouched.
    max_completion_length: int | None = 3072
    # Longest prompt seen was 2.9k tokens; a call gets min(cap, window - prompt).
    vllm_max_model_length: int | None = 7168
    mask_truncated_completions: bool = False
    # Stability: runs at 5e-6 degraded after their peak; 2e-6 did not (LOG.md).
    learning_rate: float = 2e-6
    beta: float = field(default=0.0, metadata={
        "help": "coefficient of a KL term to the model the run starts from; 0 means no "
                "term and no reference model. 0.04 kept a run from learning the task. A "
                "reference is a second copy of the weights on the GPU and one more "
                "no-gradient forward pass per micro-batch"})
    lr_scheduler_type: str = "constant_with_warmup"
    # The warm-up starts above zero (Trainer.create_scheduler): with 4 warm-up steps
    # the learning rate is 0.2, 0.4, 0.6, 0.8 of its value, then constant.
    warmup_steps: float = 4
    max_grad_norm: float = 1.0
    disable_dropout: bool = True
    vllm_importance_sampling_mode: str = "token_truncate"
    max_steps: int = 60

    # ---- memory, storage and logging: none of it changes the update -------------------
    token_budget: int = field(default=8192, metadata={
        "help": "padded tokens (rows x longest row) per backward micro-batch. On an H100 "
                "16384 was 14% slower per token, and turning gradient checkpointing off "
                "ran out of memory at 8192 (2026-11-19)"})
    bf16: bool | None = True
    model_init_kwargs: dict[str, Any] | str | None = field(
        default_factory=lambda: {"dtype": "bfloat16"})
    gradient_checkpointing: bool = True
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    # vLLM gives its memory back before the backward passes (Trainer._collect).
    vllm_enable_sleep_mode: bool = True
    vllm_gpu_memory_utilization: float = 0.35
    gradient_accumulation_steps: int = 1
    per_device_train_batch_size: int = 0
    shuffle_dataset: bool | None = False
    save_strategy: str = "steps"
    save_steps: float = 20
    save_only_model: bool = False
    optimizer_on_latest_only: bool = field(default=True, metadata={
        "help": "keep the optimizer state (32 GB) only in the newest checkpoint; earlier "
                "ones keep their weights"})
    logging_steps: float = 1
    report_to: None | str | list[str] = "none"
    log_completions: bool = False

    def __post_init__(self):
        if not self.name:
            raise ValueError("every run needs a --name: its output directory keys on it")
        if self.baseline not in BASELINES:
            raise ValueError(f"--baseline must be one of {BASELINES}")
        if self.baseline == "other_groups" and self.groups_per_step < 2:
            raise ValueError("--baseline other_groups needs at least two groups per step")
        if self.scale_rewards != "none":
            raise ValueError("advantages are not scaled in this project")
        if self.mask_truncated_completions:
            raise ValueError("truncated calls stay in the loss in this project")
        if self.eval_peers and not (self.eval_only and self.shared_machine):
            raise ValueError("--eval_peers is for an evaluation on shared machines")
        rollouts = self.groups_per_step * self.num_generations
        if self.per_device_train_batch_size not in (0, rollouts):
            raise ValueError("set --groups_per_step, not --per_device_train_batch_size")
        if self.gradient_accumulation_steps != 1:
            raise ValueError("a step's rows are accumulated inside training_step; "
                             "gradient_accumulation_steps must stay 1")
        # Before super(): GRPOConfig derives generation_batch_size from this.
        self.per_device_train_batch_size = rollouts
        self.output_dir = self.output_dir or str(base.OUT / "runs" / self.name)
        self.run_name = self.run_name or self.name
        super().__post_init__()


def advantages(rewards, baseline):
    """rewards[g][m] is a float, or None for a rollout that failed (it gets no advantage
    and is left out of every mean).
    """
    mean = lambda xs: sum(xs) / len(xs) if xs else None

    flat = [[r for r in grp if r is not None] for grp in rewards]
    out = []
    for g, grp in enumerate(rewards):
        row = []
        for m, r in enumerate(grp):
            if r is None:
                row.append(None)
                continue
            if baseline == "group":
                b = mean(flat[g])
            elif baseline == "loo":
                b = mean([x for k, x in enumerate(grp) if k != m and x is not None])
            else:
                b = mean([x for h, other in enumerate(flat) if h != g for x in other])
            row.append(0.0 if b is None else r - b)
        out.append(row)
    return out


def pack(rows, budget):
    """Micro-batches of rows, longest first, each at most `budget` padded tokens."""
    rows = sorted(rows, key=lambda r: -(len(r["prompt_ids"]) + len(r["completion_ids"])))
    batches, cur = [], []
    for r in rows:
        longest = len((cur or [r])[0]["prompt_ids"]) + len((cur or [r])[0]["completion_ids"])
        if cur and longest * (len(cur) + 1) > budget:
            batches.append(cur)
            cur = []
        cur.append(r)
    if cur:
        batches.append(cur)
    return batches


def quantile(xs, q):
    xs = sorted(xs)
    return float(xs[min(len(xs) - 1, int(q * len(xs)))]) if xs else float("nan")


def engine(vg):
    """(enqueue, poll) for Broker.serve, on the queue of the vLLM engine in `vg` (TRL's
    VLLMGeneration). Sampling settings are the ones TRL's colocated call uses."""
    from vllm import SamplingParams
    params = dict(n=1, repetition_penalty=vg.repetition_penalty, temperature=vg.temperature,
                  top_p=vg.top_p, top_k=vg.top_k, min_p=0.0 if vg.min_p is None else vg.min_p,
                  logprobs=vg.logprobs)
    params.update(vg.generation_kwargs)
    # vLLM's internal request id is "<counter>-<8 random characters>"; depending on
    # the call it reports the whole id or the counter alone, so the counter is the key.
    key = lambda rid: str(rid).split("-", 1)[0]

    def enqueue(requests):
        ids = vg.llm.enqueue([{"prompt_token_ids": i} for i, _ in requests],
                             sampling_params=[SamplingParams(**{**params, "max_tokens": m})
                                              for _, m in requests], use_tqdm=False)
        return [key(i) for i in ids]

    def poll():
        done = []
        for o in vg.llm.llm_engine.step():
            if o.finished:
                c = o.outputs[0]
                ids = list(c.token_ids)
                done.append((key(o.request_id), {
                    "ids": ids, "logprobs": [lp[t].logprob for t, lp in zip(ids, c.logprobs)],
                    "finish_reason": c.finish_reason}))
        return done
    return enqueue, poll


class Trainer(base.Trainer):
    def __init__(self, *args, env, env_cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self.env, self.env_cfg = env, env_cfg

    def create_scheduler(self, num_training_steps, optimizer=None):
        """Constant with a linear warm-up that starts above zero: step s of the warm-up
        runs at (s + 1) / (warmup_steps + 1) of the learning rate. transformers' version
        starts at exactly 0, which makes the first step a rollout with no update."""
        kind = getattr(self.args.lr_scheduler_type, "value", self.args.lr_scheduler_type)
        if self.lr_scheduler is None and kind == "constant_with_warmup":
            from torch.optim.lr_scheduler import LambdaLR
            w = int(self.args.warmup_steps)
            self.lr_scheduler = LambdaLR(self.optimizer if optimizer is None else optimizer,
                                         lambda s: min(1.0, (s + 1) / (w + 1)))
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer)

    # ---- generation -------------------------------------------------------------------
    def _collect(self, inputs):
        G = self.args.num_generations
        seeds = [inputs[i]["episode_seed"] for i in range(0, len(inputs), G)]
        assert all(inputs[i]["episode_seed"] == seeds[i // G] for i in range(len(inputs))), \
            "the rollouts of a group must be consecutive rows of one seed"
        vg = self.vllm_generation
        # TRL's own generate() puts vLLM to sleep after every call, dropping its weights
        # (level 2) and its prefix cache. A step here makes hundreds of calls, so vLLM is
        # woken once, used for all of them, and put to sleep before the backward passes.
        if self.state.global_step != self._last_loaded_step or vg._llm_weights_sleeping:
            vg.sync_weights()
            self._last_loaded_step = self.state.global_step
        if vg.enable_sleep_mode:
            vg.llm.wake_up(tags=["kv_cache"])
        try:
            return seeds, *rollout.run_groups(
                self.env, self.env_cfg, G, seeds, self.args.shared_machine, engine(vg),
                self.processing_class, self.max_completion_length,
                self.args.vllm_max_model_length, workdir=os.environ.get("SLURM_TMPDIR"))
        finally:
            if vg.enable_sleep_mode:
                vg.llm.sleep(level=2)
                vg._llm_weights_sleeping = True

    # ---- one optimizer step -----------------------------------------------------------
    def training_step(self, model, inputs, num_items_in_batch=None):
        from trl.models.utils import disable_gradient_checkpointing
        t0 = time.perf_counter()
        model.train()
        seeds, groups, rows, broker = self._collect(inputs)
        t_gen = time.perf_counter() - t0

        rewards = [[None if x is None else x["reward"] for x in grp] for grp in groups]
        adv = advantages(rewards, self.args.baseline)
        scored = sum(r is not None for grp in rewards for r in grp)
        for k, v in record(Path(self.args.output_dir), int(self.state.global_step), self.args.baseline,
                           self.args.shared_machine, self.env, seeds, groups, rows, adv, broker,
                           t_gen).items():
            self._metrics["train"][k].append(float(v))
        for r in rows:
            r["advantage"] = adv[r["group"]][r["member"]]
        # A row whose rollout has advantage exactly 0 (a group that all scored the same)
        # has zero loss and zero gradient, so its forward and backward passes are skipped,
        # unless there is a KL term (--beta), which every sampled row carries.
        live = [r for r in rows if r["advantage"] or self.beta]
        device = self.accelerator.device
        total = torch.zeros((), device=device)
        for mb in pack(live, self.args.token_budget):
            batch = self._tensors(mb, device)
            ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
            attn = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
            if self.use_vllm and self.vllm_importance_sampling_correction:
                with torch.no_grad(), disable_gradient_checkpointing(
                        self.model, self.args.gradient_checkpointing_kwargs):
                    old, _, _ = self._get_per_token_logps_and_entropies(
                        self.model, ids, attn, batch["completion_ids"].size(1))
                batch["old_per_token_logps"] = old
                batch["importance_sampling_ratio"] = self._is_ratio(
                    old, batch["sampling_per_token_logps"], batch["completion_mask"])
                self._log_mismatch(batch)
            if self.beta:
                with torch.no_grad():
                    batch["ref_per_token_logps"], _, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model, ids, attn, batch["completion_ids"].size(1))
            del ids, attn
            with self.compute_loss_context_manager():
                # _compute_loss divides by (rows in this micro-batch x max_completion_
                # length); rescaled so the step's loss is the sum over all rows divided
                # by (rollouts x max_completion_length).
                loss = self._compute_loss(model, batch) * (len(mb) / max(1, scored))
            self.accelerator.backward(loss)
            total += loss.detach()
            del batch, loss
        self._step += 1
        m = self._metrics["train"]
        m["time/train_s"].append(time.perf_counter() - t0 - t_gen)
        m["step_time"].append(time.perf_counter() - t0)
        m["rows/trained"].append(len(live))
        # The clock time lets the optimizer step and checkpointing, which happen outside
        # this function, be read off as the gap between one step's line and the next.
        print(f"  timing: generate {t_gen:.0f}s (vLLM busy {broker.busy_s:.0f}s)  "
              f"train {time.perf_counter() - t0 - t_gen:.0f}s on {len(live)} of {len(rows)} rows  "
              f"at {time.strftime('%H:%M:%S')}", flush=True)
        return total

    def _tensors(self, mb, device):
        from trl.trainer.utils import pad
        t = lambda xs, dtype=torch.long: [torch.tensor(x, dtype=dtype) for x in xs]
        pad_id = int(self._tokenizer.pad_token_id)
        p = t([r["prompt_ids"] for r in mb])
        c = t([r["completion_ids"] for r in mb])
        lp = t([[float("nan") if x is None else x for x in r["logprobs"]] for r in mb],
               torch.float32)
        return {
            "prompt_ids": pad(p, padding_value=pad_id, padding_side="left").to(device),
            "prompt_mask": pad([torch.ones_like(x) for x in p], padding_value=0,
                               padding_side="left").to(device),
            "completion_ids": pad(c, padding_value=pad_id, padding_side="right").to(device),
            "completion_mask": pad([torch.ones_like(x) for x in c], padding_value=0,
                                   padding_side="right").to(device),
            "sampling_per_token_logps": pad(lp, padding_value=0.0,
                                            padding_side="right").to(device),
            "advantages": torch.tensor([r["advantage"] for r in mb], dtype=torch.float32,
                                       device=device),
        }

    def _is_ratio(self, old, sampling, mask):
        """TRL's vLLM importance-sampling ratio (`_generate_and_score_completions`)."""
        diff = torch.nan_to_num((old - sampling) * mask, nan=0.0)
        mode = self.vllm_importance_sampling_mode
        if mode.startswith("sequence"):
            diff = diff.sum(dim=-1, keepdim=True)
        ratio = torch.exp(diff)
        lo, hi = self.vllm_importance_sampling_clip_min, self.vllm_importance_sampling_clip_max
        if mode.endswith("truncate"):
            return torch.clamp(ratio, min=lo, max=hi)
        bad = (ratio < (-math.inf if lo is None else lo)) | (ratio > (math.inf if hi is None else hi))
        return ratio.masked_fill(bad, 0.0)

    # ---- records ----------------------------------------------------------------------


def record(root, step, baseline, shared_machine, env, seeds, groups, rows, adv, broker, t_gen):
    """Write one step's rollouts (rollouts.jsonl, tokens/) under `root`, print its line,
    and return its metrics."""
    S = [x["summary"] for grp in groups for x in grp if x is not None]
    n = max(1, len(S))
    rew = [s["reward"] for s in S]
    per_call = [r["completion_tokens"] for r in rows]
    by_rollout = {}
    for r in rows:
        by_rollout.setdefault((r["group"], r["member"]), []).append(r)
    flat_adv = [a for grp in adv for a in grp if a is not None]
    m = {}
    put = m.__setitem__
    put("reward", sum(rew) / n)
    put("reward_std", (sum((x - sum(rew) / n) ** 2 for x in rew) / n) ** 0.5)
    for k, v in env.metrics(S).items():
        put(f"env/{k}", v)
    put("env/failed_rollouts", sum(x is None for grp in groups for x in grp))
    # Truncation: the cap can bind more as training lengthens the thinking.
    put("tokens/per_call_mean", sum(per_call) / max(1, len(per_call)))
    put("tokens/per_call_p90", quantile(per_call, 0.9))
    put("tokens/longest_call_p90", quantile(
        [max(r["completion_tokens"] for r in rs) for rs in by_rollout.values()], 0.9))
    put("tokens/cap_hit_calls", sum(r["hit_cap"] for r in rows) / max(1, len(rows)))
    put("tokens/cap_hit_rollouts", sum(any(r["hit_cap"] for r in rs)
                                       for rs in by_rollout.values()) / n)
    put("tokens/window_hit_calls", sum(r["hit_window"] for r in rows))
    put("tokens/longest_prompt", max((r["prompt_tokens"] for r in rows), default=0))
    put("adv/abs_mean", sum(abs(a) for a in flat_adv) / max(1, len(flat_adv)))
    put("adv/zero_groups", sum(all(not a for a in grp if a is not None) for grp in adv)
        / max(1, len(adv)))
    put("rows/total", len(rows))
    put("time/generate_s", t_gen)
    put("time/vllm_s", broker.busy_s)
    episodes, seen = [], set()
    for grp in groups:
        for x in grp:
            if x is not None and id(x["episode"]) not in seen:
                seen.add(id(x["episode"]))
                episodes.append(x["episode"])
    rec = {"step": step, "seeds": seeds, "baseline": baseline,
           "shared_machine": shared_machine,
           "rewards": [[None if x is None else x["reward"] for x in grp] for grp in groups],
           "advantages": adv,
           "summaries": [[None if x is None else x["summary"] for x in grp] for grp in groups],
           "calls": [{k: r[k] for k in ("group", "member", "prompt_tokens",
                                        "completion_tokens", "finish_reason", "hit_cap")}
                     for r in rows],
           "generate_s": round(t_gen, 1), "episodes": episodes}
    with open(root / "rollouts.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    # Each call's sampled token ids and vLLM log-probs (about 3 MB per step of 32
    # rollouts). One file per step, so a resumed run overwrites the steps it repeats.
    # Rows are in call order within a rollout; `prompt_tokens` is enough to rebuild
    # the prompt from the transcript (rl/check_rows.py).
    d = root / "tokens"
    d.mkdir(exist_ok=True)
    with open(d / f"step_{step:05d}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in (
                "group", "member", "prompt_tokens", "completion_ids", "logprobs",
                "finish_reason")}) + "\n")
    print(f"step {step}: reward {sum(rew) / n:.3f}  {env.progress(S)}  "
          f"calls {len(rows)}  tok/call {sum(per_call) / max(1, len(per_call)):.0f}  "
          f"cap hits {sum(r['hit_cap'] for r in rows)}  generate {t_gen:.0f}s", flush=True)
    return m


def code_version():
    """The commit the run's code is at, with "+uncommitted" if any tracked file under the
    code directories differs from it. A run is reproduced by its command line at this
    commit (LOG.md), so launch from committed code."""
    import subprocess
    root = HERE.parent
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True,
                                    text=True).stdout.strip()
    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    code = [p.name for p in root.iterdir() if p.is_dir() and any(p.glob("*.py")) or p.name == "jobs"]
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", *code)
    return commit + ("+uncommitted" if dirty else "")


def evaluate(cfg, env, env_cfg, tokenizer, root, eng=None):
    """Rollouts and their records with no trainer: only vLLM holds the model, awake for
    the whole run and with most of the GPU (trainer_base.vllm_engine), so a step can be
    many groups. Group i uses seed data_seed + i however the groups are split into steps.
    `eng` replaces the model's engine in the model-free test.
    """
    if (root / "rollouts.jsonl").exists():
        raise FileExistsError(f"{root} already holds rollouts; an evaluation does not "
                              "resume, so give it a new --name or remove them")
    eng = eng or engine(base.vllm_engine(cfg, tokenizer))
    peers = env.scripted(cfg.eval_peers) if cfg.eval_peers else None
    workdir = os.environ.get("SLURM_TMPDIR")
    for step in range(cfg.max_steps):
        seeds = [cfg.data_seed + step * cfg.groups_per_step + i for i in range(cfg.groups_per_step)]
        t0 = time.perf_counter()
        groups, rows, broker = rollout.run_groups(
            env, env_cfg, cfg.num_generations, seeds, cfg.shared_machine, eng, tokenizer,
            cfg.max_completion_length, cfg.vllm_max_model_length, workdir=workdir, peers=peers)
        if peers is not None:
            # What the scripted agents earn on the same seed when the model's place is
            # taken by one of them. They act on their own jobs only and their rounds do
            # not depend on the others, so the difference is what the model's rollout
            # took from them.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(8) as pool:
                alone = list(pool.map(lambda seed: env.run_episode(env_cfg, peers, seed, workdir=workdir),
                                      seeds))
            for grp, ref in zip(groups, alone):
                if grp[0] is not None:
                    with_model = grp[0]["episode"]["agents"][1:]
                    grp[0]["summary"]["harm_to_scripted"] = sum(
                        r["reward"] - w["reward"] for r, w in zip(ref["agents"][1:], with_model))
        rewards = [[None if x is None else x["reward"] for x in grp] for grp in groups]
        record(root, step, cfg.baseline, cfg.shared_machine, env, seeds, groups, rows,
               advantages(rewards, cfg.baseline), broker, time.perf_counter() - t0)


def placeholder_reward(prompts, completions, **kwargs):
    """TRL insists on a reward function; rewards come from the environment (training_step)."""
    return [0.0] * len(prompts)


def main():
    # The environment has to be known before its flags can be parsed.
    name = next((a for flag, a in zip(sys.argv, sys.argv[1:]) if flag == "--env"),
                TrainConfig.__dataclass_fields__["env"].default)
    env = importlib.import_module(name)
    cfg, env_args = base.parse(TrainConfig, env.Args)
    env_cfg = env.make(env_args, cfg.num_generations)
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    from datasets import Dataset
    from transformers import AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint

    n = cfg.max_steps * cfg.groups_per_step
    train = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": cfg.env}]] * n,   # unused: the
        # environment builds each group's request from its seed
        "episode_seed": [cfg.data_seed + i for i in range(n)]})
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")

    record = dict(
        name=cfg.name, model=cfg.model, baseline=cfg.baseline,
        shared_machine=cfg.shared_machine, env_module=cfg.env, env=asdict(env_cfg),
        groups_per_step=cfg.groups_per_step, generations=cfg.num_generations,
        steps=cfg.max_steps, lr=cfg.learning_rate, lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps, max_completion_length=cfg.max_completion_length,
        vllm_max_len=cfg.vllm_max_model_length, temperature=cfg.temperature,
        top_p=cfg.top_p, top_k=cfg.top_k, seed=cfg.seed, data_seed=cfg.data_seed,
        optim=None if cfg.eval_only else "master_adamw_fp32",
        loss_type=cfg.loss_type, beta=cfg.beta,
        mask_truncated_completions=cfg.mask_truncated_completions,
        vllm_importance_sampling_mode=cfg.vllm_importance_sampling_mode,
        token_budget=cfg.token_budget, description=cfg.description,
        job=os.environ.get("SLURM_JOB_ID"), commit=code_version(),
        command=" ".join(sys.argv[1:]), **env.describe(env_cfg))
    (root / "run.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items() if k not in ("system_prompt", "task")}),
          flush=True)

    started = time.time()
    if cfg.eval_only:
        evaluate(cfg, env, env_cfg, tokenizer, root)
        (root / "train_result.json").write_text(json.dumps(dict(
            status="complete", global_step=cfg.max_steps, wall_seconds=time.time() - started)))
        return

    trainer = Trainer(
        model=cfg.model, reward_funcs=[placeholder_reward], args=cfg, train_dataset=train,
        processing_class=tokenizer, env=env, env_cfg=env_cfg,
        callbacks=[base.RequireGradient(), base.HonourSaveSteps(),
                   *([base.OptimizerOnLatestOnly()] if cfg.optimizer_on_latest_only else [])])
    checkpoint = get_last_checkpoint(cfg.output_dir)
    print(f"Resume checkpoint: {checkpoint}", flush=True)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    (root / "train_result.json").write_text(json.dumps(dict(
        status="complete", global_step=trainer.state.global_step,
        wall_seconds=time.time() - started, metrics=result.metrics), indent=2))


if __name__ == "__main__":
    main()
