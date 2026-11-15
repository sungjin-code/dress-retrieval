#!/bin/bash
# Paired-bootstrap tables behind the numbers of the paper: test/{street,street-input,street-extra}/results.md.
# Needs the rank logs of test/metrics.py (see README.md, step 5). Run from anywhere: bash test/run_paired.sh
# OUT=<dir> writes the results.md files under <dir>/{street,street-input,street-extra}/ instead of test/ (to compare with the committed ones).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-test}
mkdir -p "$OUT/street" "$OUT/street-input" "$OUT/street-extra"
paired() { uv run python test/paired.py "$@"; }

# Table 2, 3 and the crop / mask rows (query 716, trained models, original inputs)
paired test/street/logs.csv --bins --out "$OUT/street/results.md" --pairs \
  "street4 - siglip2" "street4 - tianmu-mere" "street4 - marqo-ecom-b" "street4 - shop1" "shop1 - tianmu-mere" \
  "street4 - street4-no-jitter" "street4 - street4-no-bbox" "street4 - street4-no-pretrain" \
  "street4-crop - street4" "street4-mask - street4" "street4-mask - street4-crop" > /dev/null

# Table 4 and 5: the six input types (same OWLv2 boxes for SigLIP2, shop1 = shop-only, street4 = ours)
pairs=()
for m in siglip2 shop1 street4; do
  for v in crop mask-white mask-mean mask-blur bg-only; do pairs+=("$m|$v - $m|original"); done
  for v in mask-white mask-mean mask-blur; do pairs+=("$m|crop - $m|$v"); done
done
paired test/street-input/logs.csv --bins --out "$OUT/street-input/results.md" --pairs "${pairs[@]}" > /dev/null

# Table 3 (bounding-box loss in both stages, seeds), the ablation CIs, and the absolute R@1 per dress size of every model (Table 5, appendix)
paired test/street-extra/logs.csv test/street/logs.csv test/street-input/logs.csv --bins --abs-bins --out "$OUT/street-extra/results.md" --pairs \
  "street4 - street4-no-bbox-all" "street4 - street4-no-bbox" "shop1-no-bbox - shop1" "street4-no-bbox-all - street4-no-bbox" \
  "street4-seed43 - street4" "street4-seed44 - street4" "street4-seed43 - street4-seed44" \
  "street4 - street4-no-jitter" "street4 - street4-no-pretrain" "street4|crop - street4-crop" "street4|crop - tianmu-mere" > /dev/null

# best validation R@1 of the runs, from their training logs (when present)
vals=()
for run in shop1 shop1-nobbox street4 street4-nobbox-all street4-seed43 street4-seed44; do
  log=train/$run.log
  [ -f "$log" ] && vals+=("$run $(tr '\r' '\n' < "$log" | grep -oE 'best R@1 [0-9.]+' | tail -1 | grep -oE '[0-9.]+$')")
done
if [ ${#vals[@]} -gt 0 ]; then
  { echo; echo "best val R@1 (train/<run>.log): $(IFS=,; echo "${vals[*]}" | sed 's/,/, /g')"; } >> "$OUT/street-extra/results.md"
fi
echo "wrote $OUT/{street,street-input,street-extra}/results.md"
