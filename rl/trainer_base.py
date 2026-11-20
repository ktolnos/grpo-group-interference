"""The parts of the trainer that do not depend on the environment: AdamW over fp32 master
weights and its checkpoints, the vLLM sampling seed, three callbacks, and the command
line. `train.py` holds the configuration and the trainer; `deadlock_recovery.py`
the environment.
What Gemma needs (a turn terminator that is not its `eos_token`, parameters that get no
gradient, an embedding table too large for bitsandbytes) is kept for a second model.

Adapted from `common/grpo.py` of github.com/ktolnos/unlearning-reward-hacking (commit
b51efc4, 2026-11-06). The measurements quoted below are that project's.
"""

import contextlib
import copy
import dataclasses
import os
from pathlib import Path

import torch
from transformers import HfArgumentParser, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

OUT = Path(os.environ.get("GGI_OUT")
           or Path(os.environ.get("SCRATCH") or f"/scratch/{os.environ.get('USER', '')}") / "ggi" / "rl")

# MasterAdamW's state, saved beside optimizer.pt. The moments go into every checkpoint
# until OptimizerOnLatestOnly removes them; the fp32 weights only into the newest, since
# everywhere else the bf16 weights stand in for them.
MOMENTS = "adam_moments.pt"
MASTERS = "master_weights.pt"


def parse(config_class, *more, argv=None):
    """Read the command line into `config_class`, a GRPOConfig subclass, and into any
    further dataclasses (an environment's flags). Returns one object per class, or the
    config alone when there is only one.

    Redeclaring a field to change its default drops the inherited `--help` text with it,
    so the text is put back from GRPOConfig before the parser is built.
    """
    for f in dataclasses.fields(config_class):
        if not f.metadata:
            inherited = GRPOConfig.__dataclass_fields__.get(f.name)
            if inherited is not None:
                f.metadata = inherited.metadata
    out = HfArgumentParser((config_class, *more)).parse_args_into_dataclasses(argv)
    return out if more else out[0]


class HonourSaveSteps(TrainerCallback):
    """Save on the command line's `--save_steps`, not the resumed checkpoint's.

    On resume, transformers replaces the trainer state with the checkpoint's
    `trainer_state.json`, save interval included, and only warns that they differ.
    `on_train_begin` runs after the state is restored.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        if args.save_steps and args.save_steps >= 1 and state.save_steps != args.save_steps:
            print(f"save_steps {state.save_steps} (resumed) -> {int(args.save_steps)} (args)",
                  flush=True)
            state.save_steps = int(args.save_steps)


class OptimizerOnLatestOnly(TrainerCallback):
    """Delete the optimizer state from every checkpoint but the newest, after each save.

    A resume reads only the latest checkpoint's optimizer, while earlier checkpoints are
    kept for their weights (evaluations). The state is most of a checkpoint: 16 GB of
    bf16 moments and 16 GB of fp32 weights for Qwen3-4B, against 8 GB of model weights.
    `on_save` runs after the new checkpoint is written, so the newest keeps its own.
    """

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        ckpts = sorted(Path(args.output_dir).glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        for ckpt in ckpts[:-1]:
            for f in ("optimizer.pt", MOMENTS, MASTERS):
                (ckpt / f).unlink(missing_ok=True)


class RequireGradient(TrainerCallback):
    """Stop the run if its first `steps` optimizer steps all have a gradient of exactly 0.

    Without a KL term a step whose groups each scored the same within the group has zero
    advantage everywhere, so the loss and the gradient are 0, the step is taken, and the
    loop reports progress. One such step is normal; a run that starts with several is
    training on nothing (`env1v3` ended that way: every rollout scored 0.158).
    """

    def __init__(self, steps=3):
        self.steps = steps
        self.zeros = 0
        self.armed = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.armed or not logs or "grad_norm" not in logs:
            return
        if float(logs["grad_norm"]):
            self.armed = False
            return
        self.zeros += 1
        if self.zeros >= self.steps:
            raise RuntimeError(
                f"the first {self.zeros} optimizer steps all had grad_norm 0, so this run "
                "has trained on nothing: every group's rollouts scored the same, or no "
                "model call reached the loss")


class _MultiEos(int):
    """An int equal to any of several token ids, so that `x in [eos, pad]` is a set test.

    `in` on a list compares `needle == element`, and because this is an int *subclass*
    Python tries the element's `__eq__` first, which is this one. Reads that want a plain
    number -- `int(...)`, a tensor comparison, a fill value -- still see `primary`.
    """

    def __new__(cls, primary, allowed):
        self = super().__new__(cls, int(primary))
        self.allowed = frozenset(int(a) for a in allowed)
        return self

    def __eq__(self, other):
        try:
            return int(other) in self.allowed
        except (TypeError, ValueError):
            return NotImplemented

    def __ne__(self, other):
        equal = self.__eq__(other)
        return equal if equal is NotImplemented else not equal

    def __hash__(self):
        return int.__hash__(self)


def stop_token_ids(model, tokenizer):
    """Every token that legitimately ends a completion, and which of them the chat
    template uses.

    Three sources, because no one of them is complete: `eos_token_id` is a single id and
    on Gemma it is not the one the template emits; the model's generation config is what
    vLLM actually stops on; and the template itself is the ground truth for a chat turn.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        tokenize=True, return_dict=True)["input_ids"]
    special = set(tokenizer.all_special_ids)
    # The last *special* token, not the last token: templates commonly put a newline
    # after the terminator, and that newline is an ordinary token.
    template = next((i for i in reversed(rendered) if i in special), None)

    generated = getattr(model.generation_config, "eos_token_id", None) or []
    generated = [generated] if isinstance(generated, int) else list(generated)
    allowed = {i for i in [tokenizer.eos_token_id, template, *generated] if i is not None}
    return allowed, template


class MasterAdamW(torch.optim.Optimizer):
    """AdamW over an fp32 master copy of the trainable weights, held on the CPU.

    The GPU keeps only the bf16 parameters the forward pass reads, and their bf16
    gradients. An optimizer that writes each update straight into bf16 loses every update
    smaller than the gap between neighbouring bf16 values (|w|/128): over 20 steps at lr
    8e-6, 96.4% of Qwen3-4B's weights stayed bit-identical, and what moved was the sliver
    of near-zero weights, a biased projection of the gradient. Here updates accumulate in
    fp32 and the bf16 copy is refreshed from it every step.

    A torch Optimizer so the HF scheduler and accelerate can drive it; the scheduler's
    lr is read from `param_groups` at every step. The master copy and the moments are
    deliberately left out of `state_dict`: accelerate moves an optimizer's state to the
    GPU when it prepares it, and HF loads `optimizer.pt` onto the GPU on resume, either
    of which would put 48 GB there. `Trainer` saves and restores them on the CPU instead
    (`MOMENTS`, `MASTERS`).

    The micro-batch gradients of a step are still summed in bf16, in `p.grad`; only the
    update itself is exact.
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.masters = []
        for g in self.param_groups:
            for p in g["params"]:
                m = torch.empty(p.shape, dtype=torch.float32, device="cpu")
                m.copy_(p.detach())
                m.requires_grad_(True)  # a leaf, so assigning .grad below is legal
                self.masters.append(m)
        self.inner = torch.optim.AdamW(
            [dict(params=self.masters[i:i + len(g["params"])], weight_decay=g["weight_decay"])
             for i, g in zip(self._offsets(), self.param_groups)],
            lr=lr, betas=betas, eps=eps, fused=True)
        self.no_grad_seen = 0
        n = sum(m.numel() for m in self.masters)
        print(f"master weights: {n / 1e9:.2f}B fp32 on the CPU, AdamW moments beside them",
              flush=True)

    def _offsets(self):
        i = 0
        for g in self.param_groups:
            yield i
            i += len(g["params"])

    def _pairs(self):
        return zip((p for g in self.param_groups for p in g["params"]), self.masters)

    @torch.no_grad()
    def step(self, closure=None):
        # A parameter that requires grad but that the batch never reaches has p.grad None
        # -- Gemma 4 E2B's vision tower on text-only batches. Its master grad is cleared
        # rather than zeroed, so Adam does not keep moving it on momentum alone.
        missing = 0
        for p, m in self._pairs():
            if p.grad is None:
                m.grad = None
                missing += 1
                continue
            if m.grad is None:
                m.grad = torch.empty(p.shape, dtype=torch.float32)
            m.grad.copy_(p.grad)
        if missing and not self.no_grad_seen:
            print(f"master: {missing}/{len(self.masters)} trainable tensors got no "
                  f"gradient and are left alone", flush=True)
        self.no_grad_seen = missing
        for g, ig in zip(self.param_groups, self.inner.param_groups):
            ig["lr"] = g["lr"]
        self.inner.step()
        for p, m in self._pairs():
            p.copy_(m)

    def zero_grad(self, set_to_none=True):
        for g in self.param_groups:
            for p in g["params"]:
                p.grad = None

    def moments_state(self):
        """AdamW's moments in bf16: 16 GB for Qwen3-4B against 32 in fp32.

        bf16 keeps fp32's exponent range, so neither moment underflows, and its 3
        significant digits are ample for a saved estimate. torch casts them back to the
        fp32 masters' dtype on load.
        """
        sd = self.inner.state_dict()
        state = {i: {k: v.to(torch.bfloat16) if k != "step" and v.is_floating_point() else v
                     for k, v in st.items()} for i, st in sd["state"].items()}
        return dict(state=state, param_groups=sd["param_groups"])

    def load_state(self, moments, masters=None):
        """Restore the moments and, when the checkpoint has them, the fp32 weights.

        Without them the masters are re-read from the bf16 weights just loaded, which
        drops each weight's sub-bf16 remainder once and changes nothing after. The
        masters must be refreshed either way: they were copied when the optimizer was
        built, before HF loaded the checkpoint's weights.
        """
        self.inner.load_state_dict(moments)
        for (p, m), s in zip(self._pairs(), masters or [None] * len(self.masters)):
            m.data.copy_(p.detach() if s is None else s)
            p.data.copy_(m)


def vllm_settings(args):
    """VLLMGeneration's arguments as GRPOTrainer derives them from the config (trl 1.13,
    `GRPOTrainer.__init__`). An evaluation builds its engine from these without a trainer
    (`vllm_engine`), and `Trainer.__init__` checks that the trainer's own engine has the
    same, so the two cannot drift apart."""
    return dict(
        mode=args.vllm_mode, tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_length=args.vllm_max_model_length,
        max_num_seqs=(args.per_device_train_batch_size * args.vllm_tensor_parallel_size
                      * args.steps_per_generation),
        enable_sleep_mode=args.vllm_enable_sleep_mode, model_impl=args.vllm_model_impl,
        trust_remote_code=args.trust_remote_code, repetition_penalty=args.repetition_penalty,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
        max_completion_length=args.max_completion_length, logprobs=0,
        generation_kwargs=args.generation_kwargs or {})


@contextlib.contextmanager
def seeded_vllm(seed):
    """Add the run's seed to the one TRL gives its colocated vLLM engine while the engine
    is built. TRL seeds it with the process index alone, so every single-GPU run would
    sample from seed 0 whatever `--seed` says."""
    import trl.generation.vllm_generation as vllm_generation
    engine = vllm_generation.LLM

    def seeded(*a, **kw):
        kw["seed"] = kw.get("seed", 0) + seed
        print(f"vLLM engine seed {kw['seed']} (run seed {seed})", flush=True)
        return engine(*a, **kw)

    vllm_generation.LLM = seeded
    try:
        yield
    finally:
        vllm_generation.LLM = engine


def vllm_engine(args, tokenizer, gpu_share=0.9):
    """The colocated vLLM engine without a trainer, for evaluation: vLLM loads the model
    from `args.model` itself, stays awake, and takes `gpu_share` of the GPU, since nothing
    else is on it. Under a trainer it has 0.35 and sleeps during the backward passes.
    Sampling is the trainer's (`vllm_settings`)."""
    from accelerate import Accelerator
    from trl.generation.vllm_generation import VLLMGeneration
    model = torch.nn.Module()           # all the engine reads of the trainer's model:
    model.name_or_path = args.model     # where its weights are
    settings = {**vllm_settings(args), "enable_sleep_mode": False,
                "gpu_memory_utilization": gpu_share}
    with seeded_vllm(args.seed or 0):
        return VLLMGeneration(model=model, accelerator=Accelerator(),
                              processing_class=tokenizer, **settings)


class Trainer(GRPOTrainer):
    """GRPOTrainer with MasterAdamW as its optimizer, the run's seed in vLLM
    (`seeded_vllm`), and every stop token treated as a finished completion.

    The last is for models whose turn terminator is not their `eos_token`. TRL tests
    `ids[-1] not in (eos_token_id, pad_token_id)` to decide a completion was truncated,
    one id against a set: Gemma 4 ends a turn with `<turn|>` (106) but its `eos_token` is
    `<eos>` (1), so every rollout looked truncated and, under
    `mask_truncated_completions`, every batch was fully masked. Qwen's `eos_token` is its
    turn terminator. `train.py` builds its own rows and keeps truncated calls in the
    loss, so nothing there reads this yet; it matters to any path that goes through
    TRL's own `_generate_and_score_completions`.

    The widened id goes onto a copy of the trainer's own tokenizer view, via
    `object.__setattr__`. Both details are load-bearing. Assigning `eos_token_id` normally
    is not a write at all: `__setattr__` strips the `_id`, converts the value back to a
    token *string* and stores that, so a custom int is silently discarded and a plain
    `copy.copy` also leaks the change into the shared tokenizer -- which is the one saved
    beside the checkpoint -- because `_special_tokens_map` is shared by reference.
    """

    def __init__(self, *args, **kwargs):
        with seeded_vllm(getattr(kwargs.get("args"), "seed", 0) or 0):
            super().__init__(*args, **kwargs)
        mine = vllm_settings(self.args)
        theirs = {k: getattr(self.vllm_generation, k) for k in mine}
        if mine != theirs:
            raise RuntimeError(
                "vllm_settings no longer matches the engine GRPOTrainer builds, so an "
                "evaluation would sample differently from training: "
                + str({k: (mine[k], theirs[k]) for k in mine if mine[k] != theirs[k]}))
        if not hasattr(self, "_tokenizer"):
            raise AttributeError(
                "TRL no longer keeps the trainer's tokenizer on `_tokenizer`; point this "
                "at whatever it now reads for the eos/pad test that decides a completion "
                "was truncated, or the run silently goes back to a single stop token")
        allowed, template = stop_token_ids(self.model, self._tokenizer)
        if template is None:
            raise ValueError("no special token ends an assistant turn in this chat "
                             "template, so a finished completion cannot be told from a "
                             "truncated one")
        # A copy, so nothing outside these checks -- prompt encoding, the tokenizer saved
        # with the checkpoint -- sees an id that is not a plain number.
        self._tokenizer = copy.copy(self._tokenizer)
        object.__setattr__(self._tokenizer, "eos_token_id", _MultiEos(template, allowed))
        print(f"stop tokens {sorted(allowed)}, turn terminator {template} "
              f"({self._tokenizer.convert_ids_to_tokens(template)!r})", flush=True)

    def create_optimizer(self, model=None):
        if self.optimizer is None:
            opt_model = self.model if model is None else model
            decay = set(self.get_decay_parameter_names(opt_model))
            named = [(n, p) for n, p in opt_model.named_parameters() if p.requires_grad]
            groups = [dict(params=[p for n, p in named if n in decay],
                           weight_decay=self.args.weight_decay),
                      dict(params=[p for n, p in named if n not in decay], weight_decay=0.0)]
            self.optimizer = MasterAdamW([g for g in groups if g["params"]],
                                         lr=self.args.learning_rate,
                                         betas=(self.args.adam_beta1, self.args.adam_beta2),
                                         eps=self.args.adam_epsilon)
        return self.optimizer

    def _bare_optimizer(self):
        # accelerate wraps the optimizer once training starts (AcceleratedOptimizer), so
        # an isinstance test on self.optimizer itself fails silently.
        return getattr(self.optimizer, "optimizer", self.optimizer)

    def _save_optimizer_and_scheduler(self, output_dir):
        super()._save_optimizer_and_scheduler(output_dir)
        if not isinstance(opt := self._bare_optimizer(), MasterAdamW):
            return
        out = Path(output_dir)
        torch.save(opt.moments_state(), out / MOMENTS)
        torch.save(opt.masters, out / MASTERS)
        # The fp32 weights only matter for resuming the newest checkpoint exactly.
        for other in out.parent.glob("checkpoint-*"):
            if other != out:
                (other / MASTERS).unlink(missing_ok=True)

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        opt = self._bare_optimizer()
        if checkpoint is None or not isinstance(opt, MasterAdamW):
            return
        ckpt = Path(checkpoint)
        if not (ckpt / MOMENTS).exists():
            raise FileNotFoundError(
                f"{ckpt / MOMENTS} is missing, so resuming would restart AdamW's moments")
        load = lambda f: torch.load(ckpt / f, map_location="cpu", weights_only=True)
        masters = load(MASTERS) if (ckpt / MASTERS).exists() else None
        opt.load_state(load(MOMENTS), masters)
        print(f"AdamW moments restored from {ckpt}; fp32 weights "
              f"{'restored' if masters is not None else 'rebuilt from the bf16 checkpoint'}",
              flush=True)

    def log(self, logs, *args, **kwargs):
        """Add the GPU high-water marks since the last log (PyTorch's allocator counters,
        which include the colocated vLLM pool), to size the GPU request. Reset after each
        log, so they are per step."""
        if torch.cuda.is_available() and "loss" in logs:
            logs["gpu/peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            logs["gpu/peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            torch.cuda.reset_peak_memory_stats()
        return super().log(logs, *args, **kwargs)

    @torch.no_grad()
    def _log_mismatch(self, out):
        """How far the trainer's log-probs are from vLLM's for the same tokens, as the
        importance-sampling rules would see it. Runs use `token_truncate`; the sequence
        rule's figures are logged to show what it would have done:
          sampling/seq_masked_frac    rows with exp(sum of gaps) > C_max (zeroed by sequence_mask)
          sampling/seq_weight_lt_0.5  rows it would weight below 0.5
          sampling/token_clipped_frac tokens whose own ratio exceeds C_max (clipped by
                                      token_truncate)
        """
        old, samp = out.get("old_per_token_logps"), out.get("sampling_per_token_logps")
        if old is None or samp is None:
            return
        m = out["completion_mask"].float()
        d = torch.nan_to_num((old - samp).float(), nan=0.0) * m
        cmax = self.args.vllm_importance_sampling_clip_max or float("inf")
        seq = d.sum(-1).exp()
        metrics = self._metrics["train" if self.model.training else "eval"]
        metrics["sampling/seq_masked_frac"].append((seq > cmax).float().mean().item())
        metrics["sampling/seq_weight_lt_0.5"].append((seq < 0.5).float().mean().item())
        metrics["sampling/token_clipped_frac"].append(
            (((d.exp() > cmax).float() * m).sum() / m.sum().clamp(min=1)).item())


def freeze_parameters(model, patterns):
    """Hold every parameter whose name contains one of `patterns` (comma-separated) fixed.

    Call this after the trainer has built the model and before `train()` builds the
    optimizer, which is when HF collects the parameters that still require grad.

    Not needed with MasterAdamW; kept for Gemma under an 8-bit bitsandbytes optimizer
    (`--optim paged_adamw_8bit`, which this project does not install or use by default),
    because bitsandbytes cannot optimise a tensor with more than INT_MAX
    elements. Gemma 4's MatFormer per-layer embedding table,
    `model.language_model.embed_tokens_per_layer.weight`, is (262144, 8960) =
    2,348,810,240 elements against an INT_MAX of 2,147,483,647, so bitsandbytes' element
    count overflows to negative, the kernel gets a nonsense grid size and CUDA returns
    "Error invalid argument at line 118 in file /src/csrc/ops.cu". Three runs died there
    before job 5463304 stepped one parameter at a time and named it. Qwen3-4B has no
    tensor within range of the limit, which is why it never hit this.
    """
    pats = [x for x in patterns.split(",") if x]
    frozen = total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if any(x in name for x in pats):
            param.requires_grad_(False)
            frozen += param.numel()
            print(f"  frozen {name} {tuple(param.shape)} numel={param.numel()}", flush=True)
    print(f"froze {frozen / 1e9:.2f}B of {total / 1e9:.2f}B params "
          f"({frozen / total:.1%}); {(total - frozen) / 1e9:.2f}B trainable", flush=True)
    big = [(n, p.numel()) for n, p in model.named_parameters()
           if p.requires_grad and p.numel() > 2**31 - 1]
    assert not big, f"still trainable past INT_MAX, bitsandbytes will fail: {big}"
