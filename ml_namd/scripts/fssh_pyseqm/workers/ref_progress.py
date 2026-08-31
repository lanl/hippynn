#!/usr/bin/env python
"""Progress + ETA for the FSSH@PySEQM keto 10 ps reference (1000 IC, static round-robin).
Per traj: REF_DONE in its log => 100%; else last 'step: N' in its output .xyz => N/nsteps.
Overall % = mean over NIC. ETA from elapsed wall (earliest log mtime -> now) and overall %.
Usage: ref_progress.py <outdir> <logdir> [--nsteps 200000] [--nic 1000] [--watch SEC]
"""
import sys, os, glob, re, time, argparse
ap = argparse.ArgumentParser()
ap.add_argument("outdir"); ap.add_argument("logdir")
ap.add_argument("--nsteps", type=int, default=200000)
ap.add_argument("--nic", type=int, default=1000)
ap.add_argument("--watch", type=int, default=0)
a = ap.parse_args()
STEP_RE = re.compile(rb"step[:\s]+(\d+)")

def last_step(xyz):
    try:
        with open(xyz, "rb") as f:
            f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - 4096))
            tail = f.read()
        m = STEP_RE.findall(tail)
        return int(m[-1]) if m else 0
    except Exception:
        return 0

def snapshot():
    logs = glob.glob(os.path.join(a.logdir, "*.log"))
    done = 0
    for lg in logs:
        try:
            with open(lg, "rb") as f:
                f.seek(0, 2); f.seek(max(0, f.tell() - 2048))
                if b"REF_DONE" in f.read():
                    done += 1
        except Exception:
            pass
    xyzs = glob.glob(os.path.join(a.outdir, "*.xyz"))
    started = len(xyzs)
    frac_sum = float(done)   # completed trajs count as 1.0
    running = 0
    for xyz in xyzs:
        base = os.path.basename(xyz)[:-4]
        lg = os.path.join(a.logdir, base + ".log")
        is_done = False
        try:
            with open(lg, "rb") as f:
                f.seek(0, 2); f.seek(max(0, f.tell() - 2048))
                is_done = b"REF_DONE" in f.read()
        except Exception:
            pass
        if is_done:
            continue
        running += 1
        frac_sum += min(1.0, last_step(xyz) / a.nsteps)
    overall = frac_sum / a.nic
    # elapsed from earliest log mtime
    mtimes = [os.path.getmtime(x) for x in logs] if logs else [time.time()]
    elapsed = time.time() - min(mtimes)
    eta = (elapsed / overall - elapsed) if overall > 1e-6 else float("inf")
    def hm(s): return f"{int(s//3600)}h{int((s%3600)//60):02d}m" if s < 1e17 else "?"
    print(f"[{time.strftime('%H:%M:%S')}] done {done}/{a.nic} | running {running} | started {started}/{a.nic} "
          f"| overall {overall*100:5.1f}% | elapsed {hm(elapsed)} | ETA {hm(eta)} "
          f"| finish ~{time.strftime('%m-%d %H:%M', time.localtime(time.time()+eta)) if eta<1e17 else '?'}")

snapshot()
while a.watch:
    time.sleep(a.watch); snapshot()
