#!/usr/bin/env python3
"""Cluster-information scrub gate for the ml_namd project tree.

The public repository must be generic and independent of any author's compute
environment. This check fails (exit code 1) if any scanned text file contains a
token that would leak a specific machine: generic parallel-filesystem roots,
batch-scheduler directives, scheduler script files, or local drive paths. Run it
before every commit and in CI.

Installation-specific names (exact host/cluster names, account/user names, unix
groups or allocations, named software environments, institutional hosts) are NOT
hardcoded in this file -- publishing a denylist would publish the very strings it
exists to catch. Put those site-specific patterns in a local file next to this tool:

    ml_namd/tools/scrub_denylist.local.txt   (git-ignored; never committed)

one regex per line (see ``scrub_denylist.local.txt.example``). The gate loads it if
present and adds those patterns to the scan, so each site keeps its own private
denylist off the public record. Without it, the generic patterns below still run.

Usage:
    python check_no_cluster_info.py [ROOT] [--verbose]

ROOT defaults to the ml_namd/ directory that contains this tool. Pass --verbose to
print the offending lines, not just per-file counts.

Developer Information
---------------------
  __project__: "ml_namd"
"""
from __future__ import annotations

import os
import re
import sys

# GENERIC leak patterns -- structural tokens that identify no specific person, host,
# or allocation, so they are safe to ship publicly. Every site-specific NAME lives in
# the git-ignored local denylist (see the module docstring), never in this file.
GENERIC_LEAK_PATTERNS = [
    r"/lustre", r"/vast", r"/scratch", r"scratch\d",   # generic parallel-FS / scratch roots
    r"\b[A-Za-z]:[\\/]",                               # windows/local drive paths (D:\, C:/) -- not http://
]
# SCHEDULER tokens are a leak in normal files, but ALLOWED inside optional_hpc/, which
# holds a deliberately generic (placeholder-only) batch template. Includes scheduler
# script file extensions, so a reference to a *.slurm / *.sbatch script is caught
# outside optional_hpc/ even when the line carries no #SBATCH directive (see L-376).
SCHED_PATTERNS = [
    r"#SBATCH", r"\bmodule load\b", r"\bsbatch\b",
    r"\bsrun\b", r"\bsqueue\b", r"\bsacct\b",
    r"\.slurm\b", r"\.sbatch\b", r"\.pbs\b", r"\.lsf\b",   # scheduler script file extensions
]

SELF = os.path.realpath(__file__)
DENYLIST_NAME = "scrub_denylist.local.txt"
LOCAL_DENYLIST = os.path.join(os.path.dirname(SELF), DENYLIST_NAME)


def load_local_patterns():
    """Read site-specific regexes from the git-ignored local denylist, if present.

    One regex per line; blank lines and lines starting with ``#`` are ignored.
    """
    patterns = []
    if os.path.isfile(LOCAL_DENYLIST):
        try:
            with open(LOCAL_DENYLIST, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
        except OSError:
            pass
    return patterns


LOCAL_PATTERNS = load_local_patterns()
LEAK_RE = re.compile("|".join(GENERIC_LEAK_PATTERNS + LOCAL_PATTERNS), re.IGNORECASE)
SCHED_RE = re.compile("|".join(SCHED_PATTERNS), re.IGNORECASE)
HPC_TEMPLATE_DIR = "optional_hpc"

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}
SKIP_EXT = {
    ".pt", ".npy", ".npz", ".h5", ".png", ".jpg", ".jpeg", ".pdf", ".gz",
    ".tar", ".zip", ".so", ".pyc", ".bin", ".pptx", ".ico", ".svg",
}
# Never scan the gate's own source or the private denylist: both legitimately contain
# the patterns, so scanning them would make the gate flag itself.
SKIP_PATHS = {SELF, os.path.realpath(LOCAL_DENYLIST)}


def iter_text_files(root):
    # The project's landing README lives at the repo root (one level above ml_namd/), OUTSIDE `root`.
    # Include it so the repo's front page is scrubbed too -- it is project-owned, not upstream HIPPYNN.
    # (Its omission once let machine-specific content survive on the landing page; see L-378.)
    if os.path.basename(os.path.normpath(root)) == "ml_namd":
        repo_readme = os.path.join(os.path.dirname(os.path.normpath(root)), "README.md")
        if os.path.isfile(repo_readme):
            yield repo_readme
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.realpath(path) in SKIP_PATHS:
                continue
            if os.path.splitext(name)[1].lower() in SKIP_EXT:
                continue
            yield path


def scan(root, verbose=False):
    results = []
    total = 0
    for path in iter_text_files(root):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
        except (UnicodeDecodeError, OSError):
            continue  # not a utf-8 text file -> not our concern
        rel = os.path.relpath(path, root)
        in_hpc = HPC_TEMPLATE_DIR in os.path.realpath(path).replace("\\", "/").split("/")
        regexes = (LEAK_RE,) if in_hpc else (LEAK_RE, SCHED_RE)
        file_hits = [(n, line.rstrip()) for n, line in enumerate(lines, 1)
                     if any(rx.search(line) for rx in regexes)]
        if file_hits:
            results.append((len(file_hits), rel, file_hits))
            total += len(file_hits)
    results.sort(reverse=True, key=lambda row: row[0])
    for count, rel, file_hits in results:
        print(f"{count:4d}  {rel}")
        if verbose:
            for n, text in file_hits[:8]:
                print(f"        {n}: {text[:140]}")
            if len(file_hits) > 8:
                print(f"        ... {len(file_hits) - 8} more")
    status = "FAIL" if total else "PASS"
    local_note = f"; {len(LOCAL_PATTERNS)} local denylist pattern(s)" if LOCAL_PATTERNS else "; no local denylist"
    print(f"\n==== cluster-info scan [{status}]: {total} hit(s) in "
          f"{len(results)} file(s) under {root}{local_note} ====")
    return 1 if total else 0


def main():
    # Flagged lines may contain non-Latin1 characters (e.g. an arrow in a guide); a
    # Windows cp1252 console would otherwise crash the verbose print (see L-376).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass
    args = [a for a in sys.argv[1:] if a != "--verbose"]
    verbose = "--verbose" in sys.argv[1:]
    root = args[0] if args else os.path.dirname(os.path.dirname(SELF))
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 2
    return scan(root, verbose=verbose)


if __name__ == "__main__":
    raise SystemExit(main())
