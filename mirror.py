#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Saalim Quadri <danascape@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Mirror branches from source repositories to one or more target repositories.

Reads a YAML map (mirrors.yaml by default), keeps a bare cache clone of every
source so repeated runs only fetch new objects, then pushes the configured
branches to each target.

Usage:
  ./mirror.py                      # run everything in mirrors.yaml
  ./mirror.py -n                   # dry run (git push --dry-run)
  ./mirror.py --only billie        # only repos whose source URL contains "billie"
  ./mirror.py -j 8 -c other.yaml   # 8 repos in parallel, different config

Auth: use SSH URLs with your key loaded, or HTTPS URLs and set MIRROR_TOKEN
to a GitHub token; it is injected for https://github.com/ URLs without ever
appearing on a command line.
"""

import argparse
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml (or apt install python3-yaml)")

MODES = ("push", "force", "mirror")


@dataclass
class Target:
    url: str
    mode: str
    branches: list  # list of (src, dst) tuples
    tags: bool
    prune: bool  # delete target branches that are not in `branches`


@dataclass
class Repo:
    source: str
    targets: list = field(default_factory=list)

    @property
    def name(self):
        return self.source.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1]


def parse_branches(items, where):
    out = []
    for item in items or []:
        src, _, dst = str(item).partition(":")
        if not src:
            raise ValueError(f"{where}: empty branch name in {item!r}")
        out.append((src, dst or src))
    return out


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    defaults = cfg.get("defaults", {})
    repos = []
    for i, r in enumerate(cfg.get("repos") or []):
        where = f"repos[{i}] ({r.get('source', '?')})"
        if "source" not in r:
            raise ValueError(f"{where}: missing 'source'")
        repo = Repo(source=r["source"])
        for t in r.get("targets") or []:
            t = {"url": t} if isinstance(t, str) else t
            mode = t.get("mode", r.get("mode", defaults.get("mode", "force")))
            if mode not in MODES:
                raise ValueError(f"{where}: mode must be one of {MODES}, got {mode!r}")
            branches = parse_branches(t.get("branches", r.get("branches")), where)
            if mode != "mirror" and not branches:
                raise ValueError(f"{where}: target {t['url']} has no branches")
            tags = t.get("tags", r.get("tags", defaults.get("tags", False)))
            prune = t.get("prune", r.get("prune", defaults.get("prune", False)))
            repo.targets.append(Target(t["url"], mode, branches, bool(tags), bool(prune)))
        if not repo.targets:
            raise ValueError(f"{where}: no targets")
        repos.append(repo)
    return repos


def git_env():
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    token = os.environ.get("MIRROR_TOKEN")
    if token:
        # Passed via GIT_CONFIG_* so the token never shows up in argv or logs.
        n = int(env.get("GIT_CONFIG_COUNT", "0"))
        env[f"GIT_CONFIG_KEY_{n}"] = f"url.https://x-access-token:{token}@github.com/.insteadOf"
        env[f"GIT_CONFIG_VALUE_{n}"] = "https://github.com/"
        env["GIT_CONFIG_COUNT"] = str(n + 1)
    return env


class Runner:
    def __init__(self, cache, env, log):
        self.cache, self.env, self.log = cache, env, log

    def git(self, *args, check=True):
        r = subprocess.run(["git", "-C", str(self.cache), *args], env=self.env,
                           capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        if r.returncode and check:
            raise RuntimeError(f"git {args[0]} failed:\n{out}")
        return r.returncode, out


def sync_repo(repo, cache_root, env, dry_run):
    """Fetch the source once, then push to each target. Returns (ok, log lines)."""
    log = [f"== {repo.name}  ({repo.source})"]
    key = hashlib.sha1(repo.source.encode()).hexdigest()[:12]
    cache = cache_root / f"{repo.name}-{key}.git"
    if not cache.exists():
        subprocess.run(["git", "init", "-q", "--bare", str(cache)], check=True, env=env)
    g = Runner(cache, env, log)

    try:
        _, out = g.git("ls-remote", "--heads", repo.source)
        upstream = {line.split("refs/heads/", 1)[1] for line in out.splitlines() if "refs/heads/" in line}

        want_all = any(t.mode == "mirror" for t in repo.targets)
        want_tags = want_all or any(t.tags for t in repo.targets)
        wanted = {src for t in repo.targets for src, _ in t.branches}
        missing = wanted - upstream
        for b in sorted(missing):
            log.append(f"  ! branch {b} does not exist on source, skipping it")

        refspecs = ["+refs/heads/*:refs/heads/*"] if want_all else \
                   [f"+refs/heads/{b}:refs/heads/{b}" for b in sorted(wanted - missing)]
        if want_tags:
            refspecs.append("+refs/tags/*:refs/tags/*")
        if refspecs:
            # --prune only applies to the refspecs given, so it keeps the cache exact.
            g.git("fetch", "--quiet", "--prune", "--no-tags", repo.source, *refspecs)
    except RuntimeError as e:
        log.append(f"  FAIL fetch: {e}")
        return False, log

    ok = True
    for t in repo.targets:
        args = ["push", "--porcelain"] + (["--dry-run"] if dry_run else [])
        if t.mode == "mirror":
            # Explicit refspecs rather than --mirror so nothing outside heads/tags leaks out.
            args += ["--prune", t.url, "+refs/heads/*:refs/heads/*", "+refs/tags/*:refs/tags/*"]
        else:
            force = "+" if t.mode == "force" else ""
            specs = [f"{force}refs/heads/{s}:refs/heads/{d}" for s, d in t.branches if s not in missing]
            if t.tags:
                specs.append(f"{force}refs/tags/*:refs/tags/*")
            if t.prune:
                rc, out = g.git("ls-remote", "--heads", t.url, check=False)
                if rc:
                    ok = False
                    log.append(f"  -> {t.url}: FAIL listing branches for prune:\n{out}")
                    continue
                # Mapped branches are kept even when skipped as missing on the source.
                keep = {d for _, d in t.branches}
                existing = {line.split("refs/heads/", 1)[1] for line in out.splitlines() if "refs/heads/" in line}
                specs += [f":refs/heads/{b}" for b in sorted(existing - keep)]
            if not specs:
                log.append(f"  -> {t.url}: nothing to push")
                continue
            args += [t.url, *specs]

        rc, out = g.git(*args, check=False)
        status = "ok" if rc == 0 else "FAIL"
        ok &= rc == 0
        log.append(f"  -> {t.url} [{t.mode}] {status}")
        for line in out.splitlines():
            # porcelain lines: "<flag>\t<from>:<to>\t<summary>"; drop the noise.
            if line.startswith(("To ", "Done")) or not line.strip():
                continue
            if rc == 0 and line.startswith("="):
                continue  # already up to date
            log.append(f"       {line}")
    return ok, log


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", default=here / "mirrors.yaml", type=Path)
    p.add_argument("--cache", default=Path(os.environ.get("MIRROR_CACHE", here / ".mirror-cache")), type=Path,
                   help="directory for bare source clones (default: .mirror-cache or $MIRROR_CACHE)")
    p.add_argument("-j", "--jobs", type=int, default=4, help="repos to sync in parallel")
    p.add_argument("-n", "--dry-run", action="store_true")
    p.add_argument("--only", action="append", default=[],
                   help="only sync the repo with this exact name, or else repos whose source URL "
                        "contains this text (repeatable)")
    args = p.parse_args()

    try:
        repos = load_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as e:
        sys.exit(f"config error: {e}")
    if args.only:
        names = {r.name for r in repos}

        def selected(r):
            # An exact repo name selects just that repo; anything else matches part of the URL.
            return any(r.name == s if s in names else s in r.source for s in args.only)
        repos = [r for r in repos if selected(r)]
    if not repos:
        sys.exit("no repos selected")

    args.cache.mkdir(parents=True, exist_ok=True)
    env = git_env()
    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [(r, pool.submit(sync_repo, r, args.cache, env, args.dry_run)) for r in repos]
        for repo, fut in futures:
            ok, log = fut.result()
            print("\n".join(log), flush=True)
            if not ok:
                failed.append(repo.name)

    print(f"\n{len(repos) - len(failed)}/{len(repos)} repos synced" + (" (dry run)" if args.dry_run else ""))
    if failed:
        print("failed: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
