#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${1:-exp}"
LIGHTING="${2:-normal}"
SCENARIO="${3:-static}"
GROUND_TRUTH_M="${4:--1.0}"
NOTES="${5:-}"

WS_DIR="/home/nedas/dev_ws"
OUT_DIR="${WS_DIR}/experiment_logs"
SESSION_NAME="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUT_DIR}/bags" "${OUT_DIR}/logs"

source /opt/ros/jazzy/setup.bash
source "${WS_DIR}/install/setup.bash"

echo "[record] session=${SESSION_NAME}"
echo "[record] experiment_id=${EXP_ID} lighting=${LIGHTING} scenario=${SCENARIO} gt=${GROUND_TRUTH_M}"

cleanup() {
  if [[ -n "${LOGGER_PID:-}" ]] && kill -0 "${LOGGER_PID}" 2>/dev/null; then
    kill "${LOGGER_PID}" 2>/dev/null || true
    wait "${LOGGER_PID}" 2>/dev/null || true
  fi
  echo "[record] stopped"
}
trap cleanup EXIT INT TERM

ros2 run safety experiment_logger_node --ros-args \
  -p experiment_id:="${EXP_ID}" \
  -p lighting:="${LIGHTING}" \
  -p scenario:="${SCENARIO}" \
  -p ground_truth_m:="${GROUND_TRUTH_M}" \
  -p notes:="${NOTES}" \
  -p session_name:="${SESSION_NAME}" \
  -p output_dir:="${OUT_DIR}" \
  > "${OUT_DIR}/logs/logger_${SESSION_NAME}.log" 2>&1 &
LOGGER_PID=$!

echo "[record] logger pid=${LOGGER_PID}"
echo "[record] rosbag output=${OUT_DIR}/bags/${EXP_ID}_${SESSION_NAME}"
echo "[record] Ctrl+C to stop"

ros2 bag record -o "${OUT_DIR}/bags/${EXP_ID}_${SESSION_NAME}" \
  /camera/intensity \
  /camera/intensity_rgb \
  /camera/disparity \
  /det/persons \
  /det/viz \
  /depth/person_z \
  /depth/person_status \
  /depth/person_detected \
  /depth/depth_valid \
  /depth/in_zone \
  /depth/colormap \
  /safety/state \
  /safety/level
