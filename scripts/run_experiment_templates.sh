#!/usr/bin/env bash
set -euo pipefail

# Wrapper around scripts/run_experiment_record.sh
# Usage:
#   ./scripts/run_experiment_templates.sh 1
#   ./scripts/run_experiment_templates.sh 2 400_lux
#   ./scripts/run_experiment_templates.sh 3 400_lux 1.5
#   ./scripts/run_experiment_templates.sh all

WS_DIR="/home/nedas/dev_ws"
RUN_SCRIPT="${WS_DIR}/scripts/run_experiment_record.sh"

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  echo "[error] Missing executable: ${RUN_SCRIPT}"
  exit 1
fi

MODE="${1:-help}"
LIGHTING="${2:-400_lux}"
GROUND_TRUTH_OVERRIDE="${3:-}"

run_one() {
  local exp_id="$1"
  local scenario="$2"
  local default_gt="$3"
  local notes="$4"
  local gt="${default_gt}"

  if [[ -n "${GROUND_TRUTH_OVERRIDE}" ]]; then
    gt="${GROUND_TRUTH_OVERRIDE}"
  fi

  echo
  echo "=============================="
  echo "Starting ${exp_id}"
  echo "scenario: ${scenario}"
  echo "lighting: ${LIGHTING}"
  echo "ground_truth_m: ${gt}"
  echo "notes: ${notes}"
  echo "Stop with Ctrl+C when done."
  echo "=============================="

  "${RUN_SCRIPT}" "${exp_id}" "${LIGHTING}" "${scenario}" "${gt}" "${notes}"
}

case "${MODE}" in
  1)
    run_one \
      "exp1_two_people_similar_distance" \
      "two_people_same_distance" \
      "2.0" \
      "Du zmones stovi greta, panasiu atstumu nuo kameros. Nustatytas atstumas su rulete: 2.0 m."
    ;;
  2)
    run_one \
      "exp2_two_people_different_distance" \
      "two_people_different_distance" \
      "1.5" \
      "Du zmones stovi skirtingais atstumais nuo kameros. Artimesnis zmogus: 1.5 m (rulete)."
    ;;
  3)
    run_one \
      "exp3_partial_occlusion" \
      "partial_occlusion" \
      "1.5" \
      "Vienas zmogus dalinai uzstoja kita. Tikslinis atstumas: 1.5 m (rulete)."
    ;;
  all)
    run_one \
      "exp1_two_people_similar_distance" \
      "two_people_same_distance" \
      "2.0" \
      "Du zmones stovi greta, panasiu atstumu nuo kameros. Nustatytas atstumas su rulete: 2.0 m."

    run_one \
      "exp2_two_people_different_distance" \
      "two_people_different_distance" \
      "1.5" \
      "Du zmones stovi skirtingais atstumais nuo kameros. Artimesnis zmogus: 1.5 m (rulete)."

    run_one \
      "exp3_partial_occlusion" \
      "partial_occlusion" \
      "1.5" \
      "Vienas zmogus dalinai uzstoja kita. Tikslinis atstumas: 1.5 m (rulete)."
    ;;
  *)
    cat <<USAGE
Usage:
  ./scripts/run_experiment_templates.sh <mode> [lighting] [ground_truth_override]

Modes:
  1     Du zmones greta, panasiu atstumu
  2     Du zmones skirtingais atstumais
  3     Vienas zmogus dalinai uzstoja kita
  all   Vykdo visus 3 is eiles

Defaults:
  lighting = 400_lux
  ground_truth_m:
    mode 1 -> 2.0
    mode 2 -> 1.5
    mode 3 -> 1.5

Examples:
  ./scripts/run_experiment_templates.sh 1
  ./scripts/run_experiment_templates.sh 2 400_lux
  ./scripts/run_experiment_templates.sh all 400_lux
  ./scripts/run_experiment_templates.sh 3 400_lux 1.6
USAGE
    ;;
esac
