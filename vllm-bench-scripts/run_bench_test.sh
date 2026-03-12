#!/usr/bin/env bash
set -euo pipefail

######################################
# 基本配置
######################################

MODEL="/data/disk3/hf_models/Qwen3-235B-A22B-Thinking-2507-FP8"
HOST="127.0.0.1"
PORT="9001"

# MODEL="/data/disk8/hf_models/Qwen3-Coder-480B-A35B-Instruct-FP8"
# HOST="127.0.0.1"
# PORT="9001"

# 随机数据参数
RANDOM_INPUT_LEN=2048
RANDOM_OUTPUT_LEN=2048

# fro qwen 480b 2k/1k case
# RANDOM_INPUT_LEN=2048
# RANDOM_OUTPUT_LEN=1024

# RANDOM_INPUT_LEN=98304
# RANDOM_OUTPUT_LEN=1024

RANDOM_RANGE_RATIO=0.2

# 日志配置（自动从模型路径提取模型名）
LOGDIR="./benchmark_logs"

# LOGDIR="./vllm-fork_benchmark_logs"

MODEL_NAME=$(basename "$MODEL")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOGDIR/vllm_bench_${MODEL_NAME}_input_${RANDOM_INPUT_LEN}_output_${RANDOM_OUTPUT_LEN}_${TIMESTAMP}.log"

mkdir -p "$LOGDIR"

######################################
# 测试参数组合（主要改这里）
######################################

# 格式：num-prompts:max-concurrency
# 每行定义一对测试参数
# for 2048/2048 case
TEST_CONFIGS=(
  "11:1"
  "88:8"
  "176:16"
  "352:32"
  "400:64"
  "400:128"  
  )

# for qwen 480b 2048/1024 case
# TEST_CONFIGS=(
#   "30:1"
#   "240:8"
#   "480:16"
#   "500:32"
#   "500:64"
#   "500:128"  
#   )

# for 98304/1024 case
# TEST_CONFIGS=(
#   "8:1"
#   "16:2"
#   "32:4"
#   "48:6"
#   "64:8"  
#   )


######################################
# 固定的 bench 参数
######################################

COMMON_ARGS=(
  --backend vllm
  --model "$MODEL"
  --trust-remote-code
  --host "$HOST"
  --port "$PORT"
  --dataset-name random
  --random-input-len "$RANDOM_INPUT_LEN"
  --random-output-len "$RANDOM_OUTPUT_LEN"
  --random-range-ratio "$RANDOM_RANGE_RATIO"
  --request-rate inf
  --seed 0
  --ignore_eos
)

######################################
# 开始测试
######################################

echo "==================================================" | tee -a "$LOGFILE"
echo "VLLM BENCH MATRIX TEST START: $(date)" | tee -a "$LOGFILE"
echo "MODEL: $MODEL" | tee -a "$LOGFILE"
echo "RANDOM_INPUT_LEN: $RANDOM_INPUT_LEN" | tee -a "$LOGFILE"
echo "RANDOM_OUTPUT_LEN: $RANDOM_OUTPUT_LEN" | tee -a "$LOGFILE"
echo "RANDOM_RANGE_RATIO: $RANDOM_RANGE_RATIO" | tee -a "$LOGFILE"
echo "Total test configurations: ${#TEST_CONFIGS[@]}" | tee -a "$LOGFILE"
echo "==================================================" | tee -a "$LOGFILE"

TEST_COUNT=0
for CONFIG in "${TEST_CONFIGS[@]}"; do
  # 解析配置：num-prompts:max-concurrency
  IFS=':' read -r NUM_PROMPTS MAX_CONCURRENCY <<< "$CONFIG"
  
  TEST_COUNT=$((TEST_COUNT + 1))

  echo "" | tee -a "$LOGFILE"
  echo "==================================================" | tee -a "$LOGFILE"
  echo "TEST [$TEST_COUNT/${#TEST_CONFIGS[@]}] START: $(date)" | tee -a "$LOGFILE"
  echo "Configuration: num-prompts=$NUM_PROMPTS, max-concurrency=$MAX_CONCURRENCY" | tee -a "$LOGFILE"
  echo "==================================================" | tee -a "$LOGFILE"

  vllm bench serve \
    "${COMMON_ARGS[@]}" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    2>&1 | tee -a "$LOGFILE"

  echo "" | tee -a "$LOGFILE"
  echo "TEST [$TEST_COUNT/${#TEST_CONFIGS[@]}] END: $(date)" | tee -a "$LOGFILE"
  echo "==================================================" | tee -a "$LOGFILE"

  # 防止过快连续压服务（可选，可根据需要调整等待时间）
  sleep 5

done

echo "==================================================" | tee -a "$LOGFILE"
echo "ALL TESTS FINISHED: $(date)" | tee -a "$LOGFILE"
echo "==================================================" | tee -a "$LOGFILE"
