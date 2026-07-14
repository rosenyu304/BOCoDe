#!/usr/bin/env bash
# Move queued/running jobs between SLURM partitions, on demand.
#
# Rosen will ask for this at ANY time ("pi_faez freed up, move 8 jobs there"), so it has to be a
# one-liner, not a rebuild of the campaign.
#
# It is SAFE to cancel a running job: every run is idempotent and resumable. A moved job picks up
# from its checkpoint, not from zero (verified: cut at iteration 3 -> resumed at 3). The .npz is the
# single source of truth for "is this tuple done", so nothing is ever lost or double-counted.
#
#   ./move_jobs.sh --to pi_faez --n 8                     # move 8 jobs (prefers PENDING ones)
#   ./move_jobs.sh --to pi_faez --n 8 --from mit_preemptable
#   ./move_jobs.sh --to mit_normal_gpu --n 4 --running    # allow moving RUNNING jobs too
#   ./move_jobs.sh --list                                 # just show what is where
#
# After moving, the watchdog keeps everything consistent: it will not double-submit a tuple that is
# already queued, and it will resubmit anything that dies.
set -uo pipefail

ROOT=/orcd/data/faez/001/rosen/bocode
JOBLIST=$ROOT/joblist.tsv
SHA=$(cat "$ROOT/CAMPAIGN_SHA")

TO=""; N=0; FROM=""; ALLOW_RUNNING=0; LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --to) TO=$2; shift 2 ;;
    --n) N=$2; shift 2 ;;
    --from) FROM=$2; shift 2 ;;
    --running) ALLOW_RUNNING=1; shift ;;
    --list) LIST=1; shift ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

if [ "$LIST" = 1 ]; then
  echo "current jobs by partition/state:"
  squeue -u "$USER" -h -o "%P %T" | sort | uniq -c | sed 's/^/  /'
  exit 0
fi
[ -z "$TO" ] && { echo "need --to <partition>"; exit 1; }
[ "$N" -le 0 ] && { echo "need --n <count>"; exit 1; }

# pick candidates: PENDING first (free to move), then RUNNING only if --running
states="PD"
[ "$ALLOW_RUNNING" = 1 ] && states="PD,R"
filter=""
[ -n "$FROM" ] && filter="-p $FROM"

mapfile -t CAND < <(squeue -u "$USER" -h -t "$states" $filter -o "%i|%j|%P" | grep -v "|$TO$" | head -n "$N")
[ ${#CAND[@]} -eq 0 ] && { echo "no movable jobs found"; exit 0; }

gres_for() {  # GPU partitions need --gres; mit_normal is CPU-only
  case "$1" in mit_normal) echo "" ;; *) echo "gpu:1" ;; esac
}
time_for() {
  case "$1" in
    pi_faez)         echo 4-00:00:00 ;;
    mit_preemptable) echo 2-00:00:00 ;;
    mit_normal_gpu)  echo 6:00:00 ;;
    mit_normal)      echo 12:00:00 ;;
    *)               echo 6:00:00 ;;
  esac
}

moved=0
for c in "${CAND[@]}"; do
  jid=${c%%|*}; rest=${c#*|}; name=${rest%%|*}; oldpart=${rest##*|}
  # job name is bc_<problem>_<algo>_<seed> -> recover the tuple from joblist.tsv
  row=$(awk -F'\t' -v n="$name" 'NR>1 && "bc_"$1"_"$2"_"$3==n {print; exit}' "$JOBLIST")
  [ -z "$row" ] && { echo "  skip $name (not in joblist)"; continue; }
  IFS=$'\t' read -r prob algo seed _oldp dev iters _g <<< "$row"

  # a moved job resumes from its checkpoint; cancelling it loses nothing
  scancel "$jid" 2>/dev/null

  gres=$(gres_for "$TO"); extra=""; [ -n "$gres" ] && extra="--gres=$gres"
  newid=$(sbatch --parsable --partition="$TO" $extra \
      --cpus-per-task=4 --mem=16G --time="$(time_for "$TO")" \
      --job-name="$name" \
      --export=ALL,PROBLEM="$prob",ALGO="$algo",SEED="$seed",ITERS="$iters",SHA="$SHA",DEVICE="$dev" \
      "$ROOT/submit_job.sbatch" 2>/dev/null)
  if [ -n "$newid" ]; then
    echo "  MOVED $name : $oldpart(job $jid) -> $TO(job $newid)"
    # keep joblist.tsv in sync so the watchdog resubmits to the NEW partition if it dies
    awk -F'\t' -v OFS='\t' -v p="$prob" -v a="$algo" -v s="$seed" -v np="$TO" \
        '$1==p && $2==a && $3==s {$4=np} {print}' "$JOBLIST" > "$JOBLIST.tmp" && mv "$JOBLIST.tmp" "$JOBLIST"
    moved=$((moved+1))
  else
    echo "  FAILED to resubmit $name to $TO (it will be picked up by the watchdog)"
  fi
done
echo "moved $moved job(s) -> $TO"
