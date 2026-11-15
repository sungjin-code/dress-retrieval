#!/bin/bash
# Every training run of the paper, in order (about 4 hours on an RTX 4060): shop pre-train, the street hyperparameter search,
# the ablations, bounding-box loss removed in both stages, and two more seeds of the street stage.
# Run from anywhere: bash train/run_paper.sh        (DRY_RUN=1 only prints the commands)
# A run whose train/outputs/<name>_*/best exists is skipped, so an interrupted script can be started again;
# delete the folder of a run that was cut off in the middle first. wandb is off; logs are written to train/<name>.log.
set -euo pipefail
cd "$(dirname "$0")/.."

best() {  # newest checkpoint of <name>; "<name>_" + timestamp, so shop1 does not match shop1-nobbox
  ls -d train/outputs/"$1"_*/best 2>/dev/null | tail -1 || true
}

run() {  # run <output name> <config> [hydra overrides...]; SEED comes from the caller's environment if set
  local name=$1 config=$2
  shift 2
  if [ -n "$(best "$name")" ]; then echo "skip $name (done)"; return; fi
  echo ">> $name"
  local cmd=(uv run python train/train_pairs.py --config-name "$config" wandb.mode=disabled "$@")
  if [ -n "${DRY_RUN:-}" ]; then echo "   ${cmd[*]}"; else "${cmd[@]}" > "train/$name.log" 2>&1; fi
}

run shop1 shop1
run shop1-nobbox shop1-nobbox
# the street configs name a fixed shop1 timestamp (the machine of the paper); use the checkpoint trained here instead
SHOP1=$(best shop1); SHOP1=${SHOP1:-train/outputs/shop1_TIMESTAMP/best}
SHOP1_NOBBOX=$(best shop1-nobbox); SHOP1_NOBBOX=${SHOP1_NOBBOX:-train/outputs/shop1-nobbox_TIMESTAMP/best}

for name in street1 street2 street3 street4 street4-nojitter street4-nobbox; do
  run "$name" "$name" "init_adapter=$SHOP1"
done
run street4-nopretrain street4-nopretrain                                  # init_adapter: null (no shop pre-train)
run street4-nobbox-all street4-nobbox-all "init_adapter=$SHOP1_NOBBOX"    # no bounding-box loss in either stage
for seed in 43 44; do  # same configuration as street4 (seed 42, from .env); own run folder so test/metrics.py keeps picking street4
  SEED=$seed run "street4-seed$seed" street4 "init_adapter=$SHOP1" "hydra.run.dir=train/outputs/street4-seed${seed}_\${now:%Y-%m-%d_%H-%M-%S}"
done
echo "done"
