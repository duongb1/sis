import os
import subprocess
from pathlib import Path


def build_python_env(root):
    env = os.environ.copy()
    root = Path(root)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(root) if not pythonpath else os.pathsep.join([str(root), pythonpath])
    return env


def run_stage(name, cmd, done_path, force=False, dry_run=False, cwd=None):
    cwd = Path(cwd or Path.cwd())
    done_path = Path(done_path)
    print("\n" + "=" * 80)
    print(name)
    print(" ".join(str(item) for item in cmd), flush=True)
    if done_path.exists() and not force:
        print(f"Skip: found {done_path}")
        return
    if dry_run:
        return
    subprocess.run([str(item) for item in cmd], check=True, cwd=cwd, env=build_python_env(cwd))
