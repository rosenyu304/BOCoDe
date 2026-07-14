#!/usr/bin/env bash
# WATCHDOG — keeps the campaign alive on partitions that kill jobs at random.
#
# WHY THIS IS NOT OPTIONAL
#   mit_preemptable: our jobs are the LOWEST priority. They are killed the moment the node's
#                    owner wants it. A 48-hour job WILL be interrupted, usually many times.
#   mit_normal_gpu:  6-hour wall limit. Anything longer is killed mid-run.
#   mit_normal:      12-hour wall limit. Same.
#   pi_faez:         100h and we own it — but it is often FULL, so jobs sit in the queue.
#
#   `--requeue` alone is not enough: SLURM only requeues on preemption, not on a wall-clock
#   timeout, a node failure, or an OOM. The watchdog covers all of those, because it does not
#   care WHY a job died — it only asks: "is this tuple finished (does its .npz exist)? if not,
#   and nothing is running or queued for it, resubmit it."
#
#   Safe because a run is IDEMPOTENT and RESUMABLE: resubmitting a half-done tuple picks up
#   from its checkpoint (verified: cut at iter 3 -> resumed at 3, not 0).
#
# Run it inside tmux on the LOGIN node so it survives your ssh dropping:
#   tmux new -d -s watchdog '~/bocode/watchdog.sh'
#
set -uo pipefail

ROOT=/orcd/data/faez/001/rosen/bocode
RESULTS=$ROOT/Results
JOBLIST=$ROOT/joblist.tsv          # problem <tab> algo <tab> seed <tab> partition <tab> device <tab> iters
SHA=$(cat "$ROOT/CAMPAIGN_SHA")
INTERVAL=${INTERVAL:-300}          # 5 min
MAX_QUEUED=${MAX_QUEUED:-60}       # don't flood the scheduler
STUCK_MIN=${STUCK_MIN:-15}         # queued >15 min asking H100/H200 -> fall back to L40S

log() { echo "[$(date '+%F %T')] $*"; }

submit() {
  local prob=$1 algo=$2 seed=$3 part=$4 dev=$5 iters=$6 gres=$7
  local tl; case "$part" in
    pi_faez)         tl=4-00:00:00 ;;
    mit_preemptable) tl=2-00:00:00 ;;
    mit_normal_gpu)  tl=6:00:00 ;;
    mit_normal)      tl=12:00:00 ;;
    *)               tl=6:00:00 ;;
  esac
  local extra=""
  [ -n "$gres" ] && extra="--gres=$gres"
  sbatch --parsable --partition="$part" $extra \
    --cpus-per-task=4 --mem=16G --time="$tl" \
    --job-name="bc_${prob}_${algo}_${seed}" \
    --export=ALL,PROBLEM="$prob",ALGO="$algo",SEED="$seed",ITERS="$iters",SHA="$SHA",DEVICE="$dev" \
    "$ROOT/submit_job.sbatch" 2>/dev/null
}

while true; do
  # what is already running or queued (by job name, which encodes the tuple)
  mapfile -t ACTIVE < <(squeue -u "$USER" -h -o "%j" 2>/dev/null)
  declare -A busy=(); for j in "${ACTIVE[@]}"; do busy["$j"]=1; done
  n_queued=$(squeue -u "$USER" -h -t PD 2>/dev/null | wc -l)

  done_n=0; sub_n=0; run_n=${#ACTIVE[@]}
  while IFS=$'\t' read -r prob algo seed part dev iters gres; do
    [ -z "${prob:-}" ] && continue
    [[ "$prob" == \#* ]] && continue

    # FINISHED? the .npz is the single source of truth
    if [ -f "$RESULTS/$prob/$algo/seed$seed.npz" ]; then done_n=$((done_n+1)); continue; fi
    # already running or queued?
    [ -n "${busy[bc_${prob}_${algo}_${seed}]:-}" ] && continue
    # scheduler flood control
    [ "$n_queued" -ge "$MAX_QUEUED" ] && continue

    id=$(submit "$prob" "$algo" "$seed" "$part" "$dev" "$iters" "$gres")
    if [ -n "$id" ]; then
      log "SUBMIT $prob/$algo/seed$seed -> $part job=$id"
      sub_n=$((sub_n+1)); n_queued=$((n_queued+1))
    fi
  done < "$JOBLIST"

  # ---- GPU fallback: stuck in queue >STUCK_MIN asking for H100/H200 -> ask for L40S ----
  # (NOT on pi_faez: it only HAS h100/h200, so there is nothing to fall back to — we just wait.)
  while read -r jid part name pend; do
    [ -z "${jid:-}" ] && continue
    [ "$part" = "pi_faez" ] && continue
    mins=$(( $(date +%s) - $(date -d "$pend" +%s 2>/dev/null || date +%s) ))
    mins=$(( mins / 60 ))
    if [ "$mins" -ge "$STUCK_MIN" ]; then
      log "STUCK ${mins}m: $name on $part — resubmitting with L40S"
      scancel "$jid" 2>/dev/null
      # the next loop iteration will resubmit it; record the downgrade
      echo "$name" >> "$ROOT/l40s_fallback.txt"
    fi
  done < <(squeue -u "$USER" -h -t PD -o "%i %P %j %V" 2>/dev/null | grep -E "gpu|h100|h200" || true)

  log "done=$done_n running/queued=$run_n submitted=$sub_n"
  sleep "$INTERVAL"
done
