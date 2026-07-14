# Running your half of the BoCoDe campaign (for Rosen's colleague)

## STEP-BY-STEP (start here)

### Step 1 — get the code
```bash
ssh orcd-login
export ROOT=/orcd/data/faez/001/$USER/bocode     # YOUR dir, not Rosen's
mkdir -p $ROOT && cd $ROOT
git clone --branch dev/2026_06 https://github.com/rosenyu304/BOCoDe.git BOCoDe
git -C BOCoDe checkout --detach <CAMPAIGN_SHA>   # Rosen gives you this. It must match exactly.
git -C BOCoDe rev-parse --short HEAD             # verify
echo <CAMPAIGN_SHA> > CAMPAIGN_SHA
```

### Step 2 — environment (use `uv`, not conda)
```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$ROOT/bin sh
export UV_CACHE_DIR=$ROOT/.uv_cache              # keep it OUT of your HOME quota
$ROOT/bin/uv venv $ROOT/venv --python 3.12
$ROOT/bin/uv pip install --python $ROOT/venv/bin/python torch --index-url https://download.pytorch.org/whl/cu130
$ROOT/bin/uv pip install --python $ROOT/venv/bin/python 'numpy==1.26.4' botorch pymoo xgboost tabpfn
$ROOT/bin/uv pip install --python $ROOT/venv/bin/python 'gymnasium[mujoco]' mujoco
$ROOT/bin/uv pip install --python $ROOT/venv/bin/python -e $ROOT/BOCoDe
$ROOT/venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```
> ⚠️ **Do NOT pin torch.** ORCD's driver is 590, so the default **cu130** is correct (verified on a
> real GPU node). Only the *workstations* need cu128.
> ⚠️ **DO pin `numpy==1.26.4`.** An unpinned install gives numpy 2.5.1 on the cluster and 1.26.4
> elsewhere. Different numpy = silently different results.
> Why `uv` and not conda: `conda activate` does not work in non-interactive shells (jobs die with
> `python: command not found`), conda is what is filling the HOME **inode** quota, and unpinned conda
> resolved a different numpy on the cluster than on the workstations.

### Step 3 — copy the launcher and point it at your dir
```bash
cp $ROOT/BOCoDe/experiments/orcd/* $ROOT/
# edit ROOT= at the top of submit_job.sbatch, watchdog.sh, move_jobs.sh to YOUR $ROOT
```

### Step 4 — build YOUR shard (seeds 5–9; Rosen runs 0–4)
```bash
$ROOT/venv/bin/python make_joblist.py --seeds 5-9 --iters 1000 \
    --gpu-partition mit_preemptable --cpu-partition mit_normal --out joblist.tsv
```

> ### ⚠️ GPU PARTITION — read this or your queue will just sit there
> `mit_preemptable` has a **hard cap of 4 GPUs per user** (`gres/gpu=4`). Queue more than 4 GPU jobs
> and every one after the 4th sits in `PD` forever with reason **`QOSMaxGRESPerUser`** — it looks
> like the cluster is busy, but it is *your own quota* blocking you. `mit_normal_gpu` caps at **2**.
>
> Rosen's lab owns **`pi_faez`** (16x H100/H200, 100 h, no preemption), and **you do not have access
> to it** — so you are stuck with the 4-GPU cap. Two consequences:
>   1. Keep at most ~4 GPU jobs in flight; the watchdog's `MAX_QUEUED` should be small for you.
>   2. Push everything that does NOT need a GPU to `mit_normal` (96 CPUs, far easier to get):
>      `random_search`, `scbo`, `penalty` — anything that fits one GP PER CONSTRAINT is
>      **2x FASTER on CPU anyway** (measured), because dozens of tiny sequential GP fits are
>      launch-latency bound and never fill a GPU.
>
> Rosen runs the GPU-heavy half on `pi_faez`; your half should lean on `mit_normal`.

### Step 5 — launch (the watchdog does the rest)
```bash
tmux new -d -s watchdog "$ROOT/watchdog.sh"
tmux attach -t watchdog     # Ctrl-B D to detach
```
Expect **many** preemptions on `mit_preemptable` — you are the lowest priority there. That is normal:
the watchdog resubmits, and every run **resumes from its checkpoint**, not from zero.

### Step 6 — send results back
```bash
rsync -az $ROOT/Results/ orcd-login:/orcd/data/faez/001/rosen/bocode/Results/
```
Disjoint paths — this cannot clobber Rosen's work.

### The two rules that matter more than anything else
1. **If you hit a bug, push it to the repo and tell Rosen — never fix it locally.** Otherwise the two
   of you are silently running different science.
2. **A job that exits with code 90** means your repo is not on the campaign SHA. It *refuses* to run
   on the wrong code. That is deliberate.

---


You have MIT's public ORCD partitions (`mit_normal_gpu`, `mit_preemptable`, `mit_normal`) but
**not** `pi_faez`. That's fine — your half never needs it.

## ⚠️ TWO THINGS THAT WILL SILENTLY RUIN YOUR RESULTS — read before anything else

**1. DO NOT PIN `torch`.**
ORCD's driver is **590**, so the default **cu130** build is the correct one — verified on a real GPU
node (node4506, L40S: `torch 2.13.0+cu130, cuda_available=True`, and a real BO run completed on the
GPU). Only the *workstations* (drivers 535-560) need cu128. If you pin torch on the cluster you will
either downgrade a working stack or get `cuda_available=False` and silently run everything on CPU.

**2. IF YOU HIT A BUG, PUSH IT TO THE REPO. NEVER FIX IT LOCALLY.**
A local fix means you and Rosen are running **different science** while both believing you are running
the same campaign. Push the fix to `dev/2026_06`, tell Rosen, and **we both re-pin to the new SHA**.
This is not bureaucracy: an ad-hoc code sync already left four machines silently running
**sign-inverted objectives** (the optimizers were *maximizing* cost) for hours, and nothing detected
it. The SHA check exists so that cannot happen again — `submit_job.sbatch` **exits 90 rather than run
on the wrong code**.

---

**The one rule:** a result is only valid if we know *which code* produced it. Everything below
exists to make that true. If you hit a bug, **do not fix it locally** — push it to the repo and we
both re-pin. Otherwise we are silently running different science, which has already cost this
project a full day of invalid results.

---

## 1. Setup (once, ~15 min)

```bash
ssh orcd-login
ROOT=/orcd/data/faez/001/<your_username>/bocode      # your OWN dir, not Rosen's
mkdir -p $ROOT && cd $ROOT

# 1. code, pinned to the EXACT campaign commit (this is not optional)
SHA=<campaign_sha>                                   # Rosen gives you this
echo $SHA > CAMPAIGN_SHA
git clone --branch dev/2026_06 https://github.com/rosenyu304/BOCoDe.git BOCoDe
git -C BOCoDe checkout --detach $SHA
git -C BOCoDe rev-parse --short HEAD                 # MUST match $SHA

# 2. env
module load miniforge/25.11.0-0
conda create -y -p $ROOT/env python=3.12
source activate $ROOT/env
pip install torch                                    # ORCD driver is 590 -> the default cu130 build is correct.
                                                     # DO NOT pin torch here. (The workstations need cu128; the cluster does not.)
pip install botorch pymoo xgboost
pip install -e ./BOCoDe
python -c "import torch,bocode; print(torch.__version__, torch.cuda.is_available(), len(bocode.PROBLEM_REGISTRY))"

# 3. the launcher (copy from Rosen's dir)
cp /orcd/data/faez/001/rosen/bocode/{run_one.py,submit_job.sbatch,watchdog.sh,make_joblist.py} .
# edit ROOT= at the top of submit_job.sbatch and watchdog.sh to point at YOUR $ROOT
```

---

## 2. Your shard

The unit of work is a **(problem, algorithm, seed)** tuple, and each writes exactly one file:

```
Results/<problem>/<algorithm>/seed<N>.npz
```

So the work partitions cleanly and **cannot collide**: the output path is a pure function of the
tuple. Merging our halves is a plain `rsync` of disjoint paths.

**You take seeds 5–9. Rosen takes seeds 0–4.** (5 seeds each, 10 total.)

```bash
python make_joblist.py --seeds 5-9 --iters 1000 \
    --gpu-partition mit_preemptable --cpu-partition mit_normal \
    --out joblist.tsv
```

> Why seeds and not "you take problems A–M"? Problem cost varies by **700×** across this suite, so
> a by-name split would hand one of us 10× the work. Splitting by seed keeps the two halves
> balanced automatically.

---

## 3. Launch (the watchdog does everything)

```bash
tmux new -d -s watchdog "$ROOT/watchdog.sh"
tmux attach -t watchdog     # to watch; Ctrl-B D to detach
```

The watchdog every 5 minutes:
1. reads `joblist.tsv`;
2. skips any tuple whose `.npz` already exists (**idempotent** — safe to restart anytime);
3. skips any tuple already running/queued;
4. submits the rest, up to a queue cap.

**Why a watchdog and not just `sbatch --requeue`:** `--requeue` only covers *preemption*. It does
NOT cover a wall-clock timeout (`mit_normal_gpu` kills at 6 h, `mit_normal` at 12 h), a node
failure, or an OOM. The watchdog doesn't care *why* a job died — it only asks "is this tuple
finished? if not, resubmit." That's safe because every run is **resumable**: it picks up from its
checkpoint, not from zero (verified: cut at iteration 3 → resumed at 3).

Expect **many** preemptions on `mit_preemptable` — you are the lowest priority there. That is
normal and the watchdog absorbs it.

---

## 4. Where things run (measured — do not "optimize" this)

| Method | Device | Why |
|---|---|---|
| `single_task_gp`, `qnehvi`, `qnparego`, constrained-MO, **all TFM** | **GPU** | one large GP / a transformer: **GPU 2.7× faster** |
| `random_search`, `scbo`, `penalty` | **CPU** (`mit_normal`) | these fit **one GP per constraint** — dozens of tiny sequential fits are launch-latency-bound, and **GPU is 2.0× SLOWER**. CPU allocation is also easier to get. |

`constrained_ei` is **dropped** from the campaign (its cost scales with #constraints: 33.8 h/run on
the 88-constraint Truss72D).

---

## 4b. Moving jobs between partitions (Rosen will ask for this)

`pi_faez` is often full, then suddenly frees up. When it does, move work onto it — don't wait for
the queue to drain.

```bash
./orcd/move_jobs.sh --list                       # what is where right now
./orcd/move_jobs.sh --to pi_faez --n 8           # move 8 jobs onto pi_faez
./orcd/move_jobs.sh --to pi_faez --n 8 --from mit_preemptable
./orcd/move_jobs.sh --to mit_normal_gpu --n 4 --running   # move RUNNING jobs too
```

**Cancelling a running job is safe.** Every run is idempotent and resumable: the moved job picks up
from its checkpoint, not from zero. The `.npz` is the only source of truth for "is this tuple done",
so nothing is lost or double-counted. `move_jobs.sh` also rewrites `joblist.tsv`, so if the moved job
later dies the watchdog resubmits it to the **new** partition.

## 5. Sending results back

```bash
rsync -az $ROOT/Results/ orcd-login:/orcd/data/faez/001/rosen/bocode/Results/
```

Disjoint paths, so this **cannot** overwrite Rosen's work. Every `.npz` is stamped with the
`commit_sha`, `host` and `device` that produced it, so a run from stale code is detectable rather
than silently polluting the tables.

---

## 6. Things that will bite you

| Symptom | Cause | Fix |
|---|---|---|
| Job exits with rc=90 immediately | your repo is not on the campaign SHA | `git -C BOCoDe checkout --detach $SHA`. The job *refuses* to run on the wrong code — that is deliberate. |
| `torch.cuda.is_available()` is False | you pinned torch | Don't. On ORCD (driver 590) the default cu130 build is correct. |
| Job queued forever asking for a GPU | H100/H200 contention | The watchdog falls back to L40S after 15 min. On `pi_faez` there is no fallback — but you don't use `pi_faez`. |
| A method looks 50× slower than its peers | it fits a GP per constraint | Expected. Make sure it's on CPU, not GPU. |
| Results look absurd (hypervolume = 1e29) | you are computing HV yourself | Don't. Use `2026_07_Experiment/compute_hv.py`, which uses the **fixed, published** reference points. Per-run inferred ref points are **not comparable across algorithms**. |

**Never** compare methods on the raw trace index. A GP's iteration 0 has already spent `n_init`
evaluations; random search's has spent 1. Use `algorithms/_eval_utils.py`, which puts everything on
the **total-function-evaluation** axis.
