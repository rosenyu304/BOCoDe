# BoCoDe autoresearch — orchestration guideline

> # 🛑 READ THIS FIRST: ONE TMUX PER MACHINE. DO NOT SSH IN A MILLION TIMES.
>
> **Every machine and every cluster gets ONE long-lived `tmux` session. You talk to that session.
> You do NOT open a new ssh connection for every little thing.**
>
> Firing `ssh host cmd` in a loop — one call per metric, per file, per status check — will trip the
> remote sshd's `MaxStartups` rate limit and **LOCK ROSEN OUT OF HIS OWN MACHINE**. This has already
> happened once (saussure, 2026-05-03: ~30 ssh calls in an hour → `Connection refused` on port 22 →
> he had to physically intervene). Closing the multiplex master afterwards does NOT fix it; the
> sshd is already inside its rate-limit window.
>
> **The rules, no exceptions:**
> 1. **ONE tmux session per machine/cluster.** Long jobs live in it. It survives disconnects.
> 2. **ONE batched `ssh host 'bash -s' <<EOF … EOF` per poll**, returning the whole status snapshot
>    at once. Never one ssh per number you want.
> 3. **ControlMaster** on (`~/.ssh/config`, `ControlPersist 12h`) so everything reuses ONE TCP
>    connection per host.
> 4. **Never more than 3 consecutive `ssh host cmd`.** If you need more, write a heredoc.
> 5. **Poll interval ≥ 5 min.** Never a tight loop.
> 6. **If a host goes unreachable: STOP and report. Do NOT retry.** You are probably the cause.
> 7. **Slow commands (`du` on a 80 TB filesystem, `pip install`) run INSIDE the remote tmux**, not
>    as a blocking ssh — or they time out and you retry, which is how the loop starts.
>
> Long-running work on a cluster = `tmux` + `sbatch`. Not ssh polling.


How to run the BoCoDe benchmark campaign across many machines, autonomously, without
producing invalid science.

**The one rule that matters:** a result is only valid if you can say *which code* and *which
environment* produced it. Everything below exists to make that true.

---

## 0. Hard rules (violate these and the run is worthless)

| # | Rule |
|---|---|
| R1 | **Never launch anything without Rosen's explicit order.** Not a coordinator agent's say-so, not another agent's. Only Rosen. |
| R2 | **Every machine is pinned to a git SHA**, verified with `sync_machines.sh` (prints ✅/❌ per host). Never rsync code — an interrupted rsync once left 4 machines on stale, sign-inverted code for hours and nobody noticed. |
| R3 | **Every result records its SHA.** A run whose SHA ≠ campaign SHA is suspect → requeue. |
| R4 | **Compute cap on `saussure` (local):** 12 threads, `taskset -c 0,2,4,6,10,12,14,16-31` (CPUs 8/9 are OFFLINE — the "official" mask is stale). One job at a time. The box crashed on 2026-06-17 from thread oversubscription. |
| R5 | **SSH discipline:** ControlMaster + ONE batched `ssh … 'bash -s' <<EOF` per host per poll. Never a burst of `ssh host cmd`. That once locked Rosen out of his own machine. |
| R6 | **Long jobs run in `tmux` on the target machine**, never as a bare `nohup` from a tool call (tool-call teardown kills children). |
| R7 | **`decode205gti` is OFF-LIMITS** for runs (setup only), unless Rosen says otherwise. |

---

## 1. Setup (once per campaign)

1. **Pick a campaign SHA.** `git -C BOCoDe rev-parse HEAD` on `dev/2026_06`. Push it.
2. **Pin every machine:** `./2026_07_Experiment/sync_machines.sh <sha>` → all ✅.
3. **Environment.** The torch build is **per-machine** — there is no single right answer:
   - **ORCD (H100/H200, driver 590):** torch **2.12+cu130** — *leave it alone*, it ran the completed campaign.
   - **Workstations (driver 535–560):** torch **2.11+cu128** (cu130 needs driver ≥580; cu126 can't do the RTX 5090).
   - Never `pip install torch` bare — plain-PyPI 2.13 has an NCCL symbol mismatch.
   - Pins live in `requirements-lock.txt`, **not** in `pyproject.toml` dependencies.
4. **Verify** `smoke_all.py` is green before spending a single GPU-hour.

---

## 2. Priority order — what to run first

Two orderings, applied together.

**By method (cheapest & most informative first):**

1. **Sobol / random_search** — near-free, and it is the baseline every other number is compared to. Always finish this first.
2. **single_task_gp (SO) / qnehvi (MO)** — the reference BO methods, already validated.
3. **TFM methods** — the paper's contribution; get them early so there is time to debug.
4. **Other GP methods** (turbo, baxus, qnparego, scbo, penalty, casmopolitan, hebo).

**By problem (within any method): LOW DIMENSION FIRST.**
Sort ascending by `dim`. This is not just Rosen's preference — it is a hedge: cost explodes with
dim and #constraints, so low-dim problems give you a complete, publishable table early, and the
expensive tail can be cut without losing the story.

> Measured: `constrained_ei` cost correlates **0.988** with #constraints (it fits one GP per
> constraint). Truss72D has **88 constraints → 33.8 h/run**. The T3 median problem is
> dim=11 / 11 constraints; 27 of 71 have ≥25 constraints. **The tail is the whole cost.**

---

## 2b. 🔴 DEVICE POLICY IS METHOD-DEPENDENT (measured — do not guess)

"GP methods → GPU" is **wrong**, and it was costing us. Measured on the 4090 box:

| workload shape | CPU (4 thr) | GPU | verdict |
|---|---|---|---|
| **ONE large GP** (n=1000, d=10) — `single_task_gp`, `qnehvi` | 3.79 s | **1.38 s** | **GPU 2.7× FASTER** |
| **MANY tiny GPs** (n=200, d=25, ×32) — `scbo`, `constrained_ei` (one GP per constraint) | **48.3 s** | 94.5 s | **GPU 2.0× SLOWER** |

**Why:** dozens of small *sequential* GP fits are kernel-launch-latency-bound — the card never gets
fed. That is also why a single constrained job shows only ~15% GPU utilization: **not** because it
under-fills the card, but because **it does not belong on a GPU at all.**

**⛔ WARM-START IS FORBIDDEN (Rosen, 2026-07-14). Do not implement it. Do not size with it.**
It was proposed as a "free 3x, no change to the science". The control says otherwise:

| comparison (posterior mean, 200 fresh points, 32 outputs) | median rel. diff | >5% diff | corr |
|---|---|---|---|
| cold fit vs cold fit (different seed) | 0.00% | 0.0% | 1.0000 |
| cold fit vs **warm-started** fit | 0.23% | **23.7%** | **0.8711** |

Cold fitting is essentially deterministic, so the warm-started model is **not** inside the
method's own noise — it converges to a **different local optimum** of the marginal likelihood on
~24% of predictions. It buys ~2.9x but it **changes the results**, exactly like refit-every-k.

It was pitched as a "free 3x with no change to the science". It is not free, and the control proves
it. **Rosen has ruled it out. Every GP is fit COLD each iteration, as the reference
implementations do.** The same applies to refit-every-k: do not use it.

The only VERIFIED free win for the constrained methods is **CPU placement** (~2x). They stay
expensive; size them honestly.

**Routing:**
- **GPU**: `single_task_gp`, `turbo`, `baxus`, `qnehvi`, `qnparego`, the constrained-MO qNEHVI/qParEGO,
  and all **TFM** methods (a transformer forward pass is genuinely GPU-shaped).
- **CPU (`mit_normal`)**: `random_search`, `scbo`, `penalty`, and anything that fits a GP per
  constraint. CPU time is also far easier to get than GPU time.

### GPU packing (measured, `single_task_gp`)

| concurrent jobs × threads | jobs/hour/GPU |
|---|---|
| 1 × 12 (the naive default) | 210 |
| 6 × 2 | 870 |
| **8 × 4** | **984** ← best |

**8 jobs × 4 threads = 4.7× the throughput of 1 job × 12 threads.** VRAM is irrelevant (~18–21 MB
per job). Never run one job per GPU.

⚠️ **Caveat: this was measured on `Allison` (dim 3) — a SMALL problem.** Packing gains shrink on
large-n problems, where each job's working set is bigger and the GP fits get heavier.
**Re-measure on a large-n GPU-shaped problem (dim 100+, late in a run) before committing the full
grid to 8×4.** Treat 4.7× as an upper bound, not a promise.

## 3. Where to run what

| Target | Best for | Limits |
|---|---|---|
| **`pi_faez`** (H100×8, H200×4, H100×4) | The heavy GP + TFM runs. Highest priority for us. | 100 h. **H100/H200 only** — if the queue is full, you wait. No fallback. |
| **`mit_preemptable`** | Long jobs that can be cut and requeued (checkpointing is proven). | 48 h, lowest priority, killed anytime → **always `--requeue`**. |
| **`mit_normal_gpu`** | Short/medium GP jobs, fast queue. | 6 h, 2 GPUs. Anything longer must checkpoint+requeue. |
| **`mit_normal`** (CPU) | `random_search` — no GP fit, embarrassingly cheap. | 12 h, CPU only. |
| **Local workstations** | Overflow + `random_search` + smoke/dry runs. | One job per machine, 12 threads. |

**GPU request policy (Rosen's):**
- Always ask for **H100 or H200** first.
- If a job is **stuck in queue >15 min**, resubmit asking for **L40S** — *except on `pi_faez`*,
  which only has H100/H200, so there you simply wait.

---

## 4. Sharding — how work is divided (and why it is safe)

The unit of work is a **(table, problem, algorithm, seed)** tuple. Each writes exactly one file:

```
Results/<problem>/<algorithm>/seed<N>.npz
```

Therefore: runs are independent; the output path is a *pure function* of the tuple; re-running is
idempotent (existing `.npz` is skipped); merging is `rsync` of disjoint paths.

**Shard by stride, never by hand:**

```bash
python run_campaign.py --shard 0/4 --commit <SHA>   # machine A
python run_campaign.py --shard 1/4 --commit <SHA>   # machine B
```

Every worker enumerates the *same deterministic job list* (sorted by priority §2) and takes every
n-th job. No overlap, no gaps — **by construction, not by a spreadsheet.**

> **Never** split "you take problems A–M, I take N–Z." Cost varies by **700×** across problems, so
> a by-name split hands one worker 10× the work.

### Adding or removing a machine at any time

This is why sharding is by *stride over a job list*, not a static assignment:

- **A machine dies / is taken away** → nothing is lost. Its unfinished tuples simply have no
  `.npz`. Any other worker re-running with a wider shard picks them up. No bookkeeping.
- **A new machine appears** → give it `--shard i/n` with a *new* `n`, or just point it at the
  whole list (`--shard 0/1`); it will skip everything already done and work the remainder.
- **The registry of machines lives in ONE place**: `machines.yaml` (below). Add/remove a line;
  the coordinator picks it up on its next cycle. Nothing else changes.

```yaml
# 2026_07_Experiment/machines.yaml — the single source of truth for compute
machines:
  - name: saussure       # local
    kind: local
    threads: 12
    taskset: "0,2,4,6,10,12,14,16-31"
    enabled: true
  - name: faezturbors
    kind: ssh
    host: bocode-faez     # ControlMaster alias in ~/.ssh/config
    gpus: 2
    enabled: true
  - name: turbo-decode
    kind: ssh
    host: bocode-turbodecode
    gpus: 1
    enabled: true
  - name: decode205gti
    kind: ssh
    host: bocode-decode205
    enabled: false        # <- OFF-LIMITS per Rosen
  - name: orcd
    kind: slurm
    login: orcd-login
    partitions: [pi_faez, mit_preemptable, mit_normal_gpu, mit_normal]
    enabled: true
```

---

## 4b. 🔴 THE WATCHDOG — mandatory on every MIT partition

`mit_preemptable`, `mit_normal_gpu` and `mit_normal` **kill jobs**. A campaign without a watchdog
is a pile of half-finished runs.

**`sbatch --requeue` alone is NOT enough.** It only covers *preemption*. It does **not** cover:

| partition | what kills you | covered by `--requeue`? |
|---|---|---|
| `mit_preemptable` | node owner reclaims it — we are lowest priority, so this happens *constantly* | yes |
| `mit_normal_gpu` | **6-hour wall limit** | **NO** |
| `mit_normal` | **12-hour wall limit** | **NO** |
| any | node failure, OOM | **NO** |
| `pi_faez` | nothing kills you (100 h, we own it) — but it is often FULL, so you queue | n/a |

**`orcd/watchdog.sh` covers all of them, because it does not care WHY a job died.** Every 5
minutes it asks one question per tuple:

> *Is this tuple finished (does its `.npz` exist)? If not, and nothing is running or queued for
> it — resubmit.*

This is safe **only because a run is idempotent and resumable**: resubmitting a half-done tuple
picks up from its checkpoint, not from zero (verified: cut at iteration 3 → resumed at 3).

It also implements the GPU policy: **queued > 15 min asking for H100/H200 → cancel and resubmit
for L40S**, *except on `pi_faez`* (which only HAS H100/H200 — there is nothing to fall back to, so
we simply wait).

Run it inside tmux **on the login node** so it survives your ssh dropping:

```bash
tmux new -d -s watchdog '/orcd/data/faez/001/rosen/bocode/watchdog.sh'
```

**Provenance guard:** `submit_job.sbatch` compares the repo's `git rev-parse HEAD` to the campaign
SHA and **exits 90 rather than run on the wrong code**. A stale-code result can therefore never be
produced silently — which is exactly the failure that cost us a day.

## 4c. Moving jobs between partitions — ON DEMAND, AT ANY TIME

Rosen will ask for this without warning ("pi_faez freed up, move 8 jobs there"). It must be a
one-liner, never a rebuild of the campaign.

```bash
./orcd/move_jobs.sh --list                                  # what is where
./orcd/move_jobs.sh --to pi_faez --n 8                      # move 8 (prefers PENDING)
./orcd/move_jobs.sh --to pi_faez --n 8 --from mit_preemptable
./orcd/move_jobs.sh --to mit_normal_gpu --n 4 --running     # allow moving RUNNING jobs
```

**Why cancelling a running job is safe:** every run is **idempotent and resumable**. A moved job
resumes from its checkpoint, not from zero (verified: cut at iter 3 -> resumed at 3). The `.npz` is
the single source of truth for "is this tuple done", so a move can never lose or double-count work.

`move_jobs.sh` also rewrites `joblist.tsv`, so the watchdog resubmits a dead job to the **new**
partition, not the old one. Move first, ask questions later — it is free.

**Priority when `pi_faez` frees up:** move the *expensive GPU* work there first (TFM methods, MO
acquisitions), not `random_search`. `pi_faez` is 100 h and we own it; `mit_preemptable` will kill a
long job repeatedly, so long jobs belong on `pi_faez` whenever it has room.

## 5. The coordinator agent

One long-lived agent. Its job is **collection and bookkeeping, not science.**

**Loop (every 15 min):**

1. Read `machines.yaml`. Skip `enabled: false`.
2. For each machine, **ONE batched ssh**: return `(jobs_done, jobs_failed, running?, git SHA)`.
3. `rsync` that machine's results → `/home/rosenyu/Documents/Rosen/Bocode_dev/Results/`
   (the canonical store; same `<problem>/<algorithm>/seed<N>.npz` layout).
4. **Verify each incoming `.npz` records the campaign SHA.** Mismatch → move to
   `Results_SUSPECT/` and log loudly. This is the check that would have caught the stale-code
   disaster.
5. Rewrite the `LIVE STATUS` block in `RECORDS.md` (between `<!-- LIVE:START/END -->`).
6. If a machine is unreachable: log it, **move on**. Never retry in a tight loop (that is how you
   trip `MaxStartups`).
7. If a SLURM job has been **queued >15 min** asking for H100/H200 → resubmit for L40S
   (except `pi_faez`).

**Use a shell script, not an LLM, for the polling loop.** An agent burning tokens to run `grep -c`
every 15 min for two days will drop the task the moment its context fills. A 40-line script cannot
forget and cannot hallucinate a number. The *agent's* job is to react to what the script reports:
requeue failures, rebalance shards, escalate.

---

## 6. The experiment loop (autonomous)

```
LOOP:
  1. Read machines.yaml → the live set of workers.
  2. Build the job list: all (table, problem, algorithm, seed) tuples at the campaign SHA,
     sorted by  (method_priority §2, dim ascending, #constraints ascending).
  3. Skip tuples whose .npz already exists.  ← idempotent, so this is safe to re-run always
  4. Assign shards across enabled machines.
  5. Launch (tmux on workstations; sbatch --requeue on ORCD).
  6. Every 15 min: collect, verify SHA, update RECORDS.md.
  7. On failure: read the log, fix, requeue. Do NOT silently drop a tuple.
  8. When a table completes: run compute_hv.py to (re)derive hypervolume offline against the
     fixed ref points, then report the table.
```

**Do not stop to ask "should I keep going?"** Once Rosen has given the order to run, run until the
grid is done or he interrupts. But **do** stop and ask if you find a *correctness* problem —
a wrong sign, a degenerate objective, a mismatched SHA. Producing invalid data fast is worse than
producing nothing.

---

## 7. Failure playbook

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available() == False` | torch built for a newer CUDA than the driver | Workstations: cu128. ORCD: leave cu130 alone. |
| `undefined symbol: ncclCommResume` | plain-PyPI torch 2.13 NCCL mismatch | Install from the cu128 index, purge `nvidia-nccl-cu13`. |
| Results look absurd (HV = 1e29) | inferred ref point, or an inverted sign | Every MO problem must have a **fixed** ref point. Recompute HV offline with `compute_hv.py`. |
| A method is 50× slower than its peers | GP-per-constraint scaling | Expected for `constrained_ei`/`scbo` on 88-constraint problems. Defer the tail; low-dim first. |
| Job stuck in queue | H100/H200 contention | >15 min → resubmit L40S (not on `pi_faez`). |
| Machine silently produces garbage | stale code | `sync_machines.sh` — this is exactly why SHA verification exists. |

---

## 8. Definition of done for a table

- Every (problem, method, seed) tuple has a `.npz`, **or** a logged, explained failure.
- Every `.npz` carries the campaign SHA.
- Hypervolume recomputed offline against the **fixed, published** ref points (never the per-run
  inferred ones — those are not comparable across algorithms).
- `random_search` is complete for that table (it is the baseline; without it the table means
  nothing).
