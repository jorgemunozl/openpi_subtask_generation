#!/usr/bin/env bash
set -euo pipefail

# Fill these paths and values before running.
REPO_ID="NONHUMAN-RESEARCH/test-general-idx"
GENERAL_TASKS_JSON="/abs/path/to/general_tasks.parquet"
TASKS_JSON="/abs/path/to/tasks.parquet"
EXP_NAME="pi05_highlow_run"

# Optional: map dataset keys to OpenPI expected keys (images/state/actions).
# Leave empty to skip repack.
REPACK_MAPPING_JSON=""

# Optional: column names inside parquet files.
GENERAL_TASKS_COLUMN="general_task_index"
TASKS_COLUMN="task_index"

# Optional: initialize weights.
WEIGHT_TYPE="none" # none|checkpoint|paligemma
WEIGHT_PATH=""     # required if WEIGHT_TYPE=checkpoint


PYTHON_BIN="uv run"
SCRIPT_PATH="/home/jorge/project/openpi_subtask_generation/scripts/train_pi05_highlow.py"

ARGS=(
  --repo-id "${REPO_ID}"
  --general-tasks-path "${GENERAL_TASKS_JSON}"
  --tasks-path "${TASKS_JSON}"
  --exp-name "${EXP_NAME}"
  --batch-size 4
  --num-train-steps 1000
  --action-dim 32
  --action-horizon 50
  --max-token-len 200
  --seed 42
  --fsdp-devices 1
  --log-interval 100
  --save-interval 1000
  --keep-period 1000
  --weight-type "${WEIGHT_TYPE}"
)

if [[ -n "${GENERAL_TASKS_COLUMN}" ]]; then
  ARGS+=(--general-tasks-column "${GENERAL_TASKS_COLUMN}")
fi

if [[ -n "${TASKS_COLUMN}" ]]; then
  ARGS+=(--tasks-column "${TASKS_COLUMN}")
fi

if [[ -n "${WEIGHT_PATH}" ]]; then
  ARGS+=(--weight-path "${WEIGHT_PATH}")
fi

if [[ -n "${REPACK_MAPPING_JSON}" ]]; then
  ARGS+=(--repack-mapping-path "${REPACK_MAPPING_JSON}")
fi

exec "${PYTHON_BIN}" "${SCRIPT_PATH}" "${ARGS[@]}"
