#!/usr/bin/env bash
# chmod +x run_validations.sh && ./run_validations.sh -- Gives permission and runs immediately after, remember this script is in the `scripts` directory.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

OUT="$ROOT/validation_output"
mkdir -p "$OUT" "$ROOT/figures"

export PYTHONPATH="$ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

python -m src.validations.validation_dqn_cartpole 2>&1 | tee "$OUT/cartpole_validation.txt"
cartpole_status=${PIPESTATUS[0]}

python -m src.validations.validation_dqn_pacman 2>&1 | tee "$OUT/pacman_validation.txt"
pac_status=${PIPESTATUS[0]}

echo CartPole exit code: $cartpole_status | MsPacman exit code: $pac_status
echo "Figures: $ROOT/figures | Text output: $OUT"
exit $(( cp_status || pac_status ))