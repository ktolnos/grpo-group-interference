"""NeMo-RL's CodeEnvironment ("code" environment, a Python tool): one long-lived Ray worker
runs the code of many rollouts in turn with exec(), all sharing one globals dict
(`self.sandbox`, holding the builtins). Its guard blocks `import os` and confines open()
to the rollout's working directory.

The CodeExecutionWorker class is taken unmodified from nemo_rl/environments/code_environment.py
(NeMo-RL 1fcd48d); ray, torch and the other nemo_rl modules are replaced by empty stand-ins so it runs
in this process.
Usage: python nemo_rl_code_env_test.py <path to code_environment.py>
"""

import importlib.util
import sys
import tempfile
import types

# stand-ins for ray, torch and the rest of nemo_rl, which the class does not use
for name in ["ray", "torch", "nemo_rl", "nemo_rl.data", "nemo_rl.data.interfaces", "nemo_rl.distributed",
             "nemo_rl.distributed.batched_data_dict", "nemo_rl.distributed.virtual_cluster",
             "nemo_rl.environments", "nemo_rl.environments.interfaces", "nemo_rl.environments.utils"]:
    m = types.ModuleType(name)
    m.__getattr__ = lambda attr: type(attr, (), {})
    sys.modules[name] = m
sys.modules["ray"].remote = lambda c: c

spec = importlib.util.spec_from_file_location("code_environment", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
CodeExecutionWorker = mod.CodeExecutionWorker


def rollout(worker, code):
    meta = {"context": {}, "working_dir": tempfile.mkdtemp()}
    obs, _, _ = worker.execute([f"<code>{code}</code>"], [meta])
    return obs[0]["content"].strip().replace("\n", " ")[:200]


w = CodeExecutionWorker()
print("rollout 1, honest:          ", rollout(w, "sum([1, 2, 3])"))
print("rollout 2, `import os`:     ", rollout(w, "import os\nos.getpid()"))
print("rollout 3, through open():  ", rollout(w, "o = __builtins__['open'].__func__.__globals__['os']\n(o.getpid(), o.getcwd())"))
print("rollout 4, overwrites sum:  ", rollout(w, "__builtins__['sum'] = lambda *a, **k: 0\n'done'"))
print("rollout 5, honest, later:   ", rollout(w, "sum([1, 2, 3])"))
print("rollout 6, honest, later:   ", rollout(w, "x = [3, 1, 2]\nsorted(x)[-1] + sum(x)"))
