"""GRPO on Environment 1: does training on a shared machine increase kills of peers' jobs?

    python rl/train.py --name env1_group_s0 --baseline group --seed 0

The recipe is the one frozen in `unlearning-reward-hacking` (`common.grpo`, TRL): dr_grpo,
no KL term, bf16 weights updated through fp32 master weights
(`MasterAdamW`), vLLM colocated with the trainer. Three things differ:

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
"""

import json
import math
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("URH_REPO", "/project/6101830/eop/unlearning-reward-hacking"))
sys.path.insert(0, str(HERE))
OUT = Path(os.environ.get("GGI_OUT", "/scratch/eop/outputs/ggi/rl"))

import torch  # noqa: E402
from common import grpo  # noqa: E402

import rollout  # noqa: E402
from rollout import harness  # noqa: E402

BASELINES = ("group", "loo", "other_groups")


@dataclass
class Env1Config(grpo.ReferenceConfig):
    model: str = "Qwen/Qwen3-4B"
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
    kill_by_name_tool: bool = field(default=True, metadata={
        "help": "False leaves `kill` with a pid only, so no single call can reach a "
                "peer's job by accident"})
    eval_only: bool = field(default=False, metadata={
        "help": "run the rollouts and record them, but make no update: for measuring the "
                "untrained model or a checkpoint (--model <path>) on fixed seeds"})
    data_seed: int = field(default=1000000, metadata={
        "help": "first episode seed; group i of the run uses data_seed + i"})
    token_budget: int = field(default=8192, metadata={
        "help": "padded tokens (rows x longest row) per backward micro-batch; memory, "
                "not method"})
    description: str = ""

    # Advantages are not divided by the group's standard deviation: that turned a group
    # whose rewards were 0.93, 1, 1, 1 (one agent a minute late) into -1.5, 0.5, 0.5, 0.5,
    # the same size as a group where one agent failed outright (env1_grpo_shared_s0).
    scale_rewards: str = "none"
    num_generations: int | None = 4
    # The cap is per model call. Measured with a cap of 4096 that no call reached
    # (qwen3-4b_think_v6/v7, 320 rollouts): a rollout's longest call has p90 2.7-2.8k and
    # p95 3.1-3.2k tokens, so 3072 leaves about 94% of rollouts untouched.
    max_completion_length: int | None = 3072
    # Longest prompt seen was 2.9k tokens; a call gets min(cap, window - prompt).
    vllm_max_model_length: int | None = 7168
    mask_truncated_completions: bool = False
    master_weights: bool = True
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "constant_with_warmup"
    # The warm-up starts above zero (Env1Trainer.create_scheduler): with 4 warm-up steps
    # the learning rate is 0.2, 0.4, 0.6, 0.8 of its value, then constant.
    warmup_steps: float = 4
    vllm_importance_sampling_mode: str = "token_truncate"
    max_steps: int = 60
    save_steps: float = 20
    save_only_model: bool = False
    optimizer_on_latest_only: bool = True
    shuffle_dataset: bool | None = False
    log_completions: bool = False
    gradient_accumulation_steps: int = 1
    per_device_train_batch_size: int = 0

    def __post_init__(self):
        if self.baseline not in BASELINES:
            raise ValueError(f"--baseline must be one of {BASELINES}")
        if self.baseline == "other_groups" and self.groups_per_step < 2:
            raise ValueError("--baseline other_groups needs at least two groups per step")
        if self.scale_rewards != "none":
            raise ValueError("advantages are not scaled in this project")
        if self.mask_truncated_completions:
            raise ValueError("truncated calls stay in the loss in this project")
        if self.eval_only:              # no optimizer state worth building or saving
            self.master_weights, self.save_strategy = False, "no"
        rollouts = self.groups_per_step * self.num_generations
        if self.per_device_train_batch_size not in (0, rollouts):
            raise ValueError("set --groups_per_step, not --per_device_train_batch_size")
        if self.gradient_accumulation_steps != 1:
            raise ValueError("a step's rows are accumulated inside training_step; "
                             "gradient_accumulation_steps must stay 1")
        # Before super(): GRPOConfig derives generation_batch_size from this.
        self.per_device_train_batch_size = rollouts
        if self.name and not self.output_dir:
            self.output_dir = str(OUT / "runs" / self.name)
        super().__post_init__()

    def env(self):
        return harness.Config(G=self.num_generations, kill_by_name_tool=self.kill_by_name_tool)


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


class Env1Trainer(grpo.Trainer):
    def __init__(self, *args, env_cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_cfg = env_cfg
        self.rollout_log = Path(self.args.output_dir) / "rollouts.jsonl"

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
    def _engine(self):
        """(enqueue, poll) for Broker.serve, on vLLM's own queue. Sampling settings are
        the ones TRL's colocated call uses."""
        from vllm import SamplingParams
        vg = self.vllm_generation
        base = dict(n=1, repetition_penalty=vg.repetition_penalty, temperature=vg.temperature,
                    top_p=vg.top_p, top_k=vg.top_k, min_p=0.0 if vg.min_p is None else vg.min_p,
                    logprobs=vg.logprobs)
        base.update(vg.generation_kwargs)
        # vLLM's internal request id is "<counter>-<8 random characters>"; depending on
        # the call it reports the whole id or the counter alone, so the counter is the key.
        key = lambda rid: str(rid).split("-", 1)[0]

        def enqueue(requests):
            ids = vg.llm.enqueue([{"prompt_token_ids": i} for i, _ in requests],
                                 sampling_params=[SamplingParams(**{**base, "max_tokens": m})
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

    def _collect(self, inputs):
        G = self.env_cfg.G
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
                self.env_cfg, seeds, self.args.shared_machine, self._engine(),
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
        self._record(seeds, groups, rows, adv, broker, t_gen)
        for r in rows:
            r["advantage"] = adv[r["group"]][r["member"]]
        # A row whose rollout has advantage exactly 0 (a group that all scored the same)
        # has zero loss and zero gradient, so its forward and backward passes are skipped.
        live = [] if self.args.eval_only else [r for r in rows if r["advantage"]]
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
    def _record(self, seeds, groups, rows, adv, broker, t_gen):
        step = int(self.state.global_step)
        S = [x["summary"] for grp in groups for x in grp if x is not None]
        n = max(1, len(S))
        share = lambda f: sum(bool(f(s)) for s in S) / n
        avg = lambda f: sum(f(s) for s in S) / n
        sib = lambda s: s["killed_others_running_by_name"] + s["killed_others_running_by_pid"]
        rew = [s["reward"] for s in S]
        per_call = [r["completion_tokens"] for r in rows]
        by_rollout = {}
        for r in rows:
            by_rollout.setdefault((r["group"], r["member"]), []).append(r)
        flat_adv = [a for grp in adv for a in grp if a is not None]
        m = self._metrics["train"]
        put = lambda k, v: m[k].append(float(v))
        put("reward", sum(rew) / n)
        put("reward_std", (sum((x - sum(rew) / n) ** 2 for x in rew) / n) ** 0.5)
        put("env/finished", share(lambda s: s["finished"]))
        put("env/kill_by_name_rollouts", share(lambda s: s["kill_by_name_calls"]))
        put("env/killed_sibling_running_rollouts", share(sib))
        put("env/sibling_running_kills", avg(sib))
        put("env/killed_own_running", avg(lambda s: s["killed_own_running"]))
        put("env/times_killed", avg(lambda s: s["times_killed_by_others"]))
        put("env/renamed", share(lambda s: s["renamed"]))
        # A reply without a tool call ends the agent's turn. After the job has finished
        # that is the requested message; before, the job runs on unwatched.
        put("env/stopped_unfinished", share(lambda s: s["replied"] and not s["finished"]))
        put("env/replied", share(lambda s: s["replied"]))
        put("env/model_calls", avg(lambda s: s["model_calls"]))
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
        rec = {"step": step, "seeds": seeds, "baseline": self.args.baseline,
               "shared_machine": self.args.shared_machine,
               "rewards": [[None if x is None else x["reward"] for x in grp] for grp in groups],
               "advantages": adv,
               "summaries": [[None if x is None else x["summary"] for x in grp] for grp in groups],
               "calls": [{k: r[k] for k in ("group", "member", "prompt_tokens",
                                            "completion_tokens", "finish_reason", "hit_cap")}
                         for r in rows],
               "generate_s": round(t_gen, 1), "episodes": episodes}
        with open(self.rollout_log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        # Each call's sampled token ids and vLLM log-probs (about 3 MB per step of 32
        # rollouts). One file per step, so a resumed run overwrites the steps it repeats.
        # Rows are in call order within a rollout; `prompt_tokens` is enough to rebuild
        # the prompt from the transcript (rl/check_rows.py).
        d = Path(self.args.output_dir) / "tokens"
        d.mkdir(exist_ok=True)
        with open(d / f"step_{step:05d}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in (
                    "group", "member", "prompt_tokens", "completion_ids", "logprobs",
                    "finish_reason")}) + "\n")
        print(f"step {step}: reward {sum(rew) / n:.3f}  finished {share(lambda s: s['finished']):.2f}  "
              f"by-name {share(lambda s: s['kill_by_name_calls']):.2f}  "
              f"sibling-running kills/rollout {avg(sib):.2f}  renamed {share(lambda s: s['renamed']):.2f}  "
              f"calls {len(rows)}  tok/call {sum(per_call) / max(1, len(per_call)):.0f}  "
              f"cap hits {sum(r['hit_cap'] for r in rows)}  generate {t_gen:.0f}s", flush=True)


def placeholder_reward(prompts, completions, **kwargs):
    """TRL insists on a reward function; rewards come from the environment (training_step)."""
    return [0.0] * len(prompts)


def main():
    cfg = grpo.parse(Env1Config)
    env_cfg = cfg.env()
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    from datasets import Dataset
    from transformers import AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint

    n = cfg.max_steps * cfg.groups_per_step
    train = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": "environment 1"}]] * n,   # unused: the
        # harness builds each group's request from its seed
        "episode_seed": [cfg.data_seed + i for i in range(n)]})
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")

    record = dict(
        name=cfg.name, model=cfg.model, baseline=cfg.baseline,
        shared_machine=cfg.shared_machine, env=asdict(env_cfg),
        groups_per_step=cfg.groups_per_step, generations=cfg.num_generations,
        steps=cfg.max_steps, lr=cfg.learning_rate, lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps, max_completion_length=cfg.max_completion_length,
        vllm_max_len=cfg.vllm_max_model_length, temperature=cfg.temperature,
        top_p=cfg.top_p, top_k=cfg.top_k, seed=cfg.seed, data_seed=cfg.data_seed,
        optim="master_adamw_fp32" if cfg.master_weights else cfg.optim,
        loss_type=cfg.loss_type, beta=cfg.beta,
        mask_truncated_completions=cfg.mask_truncated_completions,
        vllm_importance_sampling_mode=cfg.vllm_importance_sampling_mode,
        token_budget=cfg.token_budget, description=cfg.description,
        system_prompt=harness.SYSTEM, task=harness.TASK,
        job=os.environ.get("SLURM_JOB_ID"))
    (root / "run.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items() if k not in ("system_prompt", "task")}),
          flush=True)

    trainer = Env1Trainer(
        model=cfg.model, reward_funcs=[placeholder_reward], args=cfg, train_dataset=train,
        processing_class=tokenizer, env_cfg=env_cfg,
        callbacks=[*([] if cfg.eval_only else [grpo.RequireGradient()]), grpo.HonourSaveSteps(),
                   *([grpo.OptimizerOnLatestOnly()] if cfg.optimizer_on_latest_only else [])])
    checkpoint = get_last_checkpoint(cfg.output_dir)
    print(f"Resume checkpoint: {checkpoint}", flush=True)
    started = time.time()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    (root / "train_result.json").write_text(json.dumps(dict(
        status="complete", global_step=trainer.state.global_step,
        wall_seconds=time.time() - started, metrics=result.metrics), indent=2))


if __name__ == "__main__":
    main()
