#!/bin/bash
# Stage2 训练脚本（Speech-Driven Vertex Prediction），支持 vocaset / BIWI / multidataset
set -e
export OMP_NUM_THREADS=10
export KMP_INIT_AT_FORK=FALSE
export PYTHONPATH=/home/caizhuoqiang/Code/audio_driven_baseline/CodeTalker:$PYTHONPATH

# 环境激活：优先 conda，其次 venv
if conda env list | grep -q "^codetalker "; then
    echo "使用conda环境: codetalker"
    conda activate codetalker
else
    echo "使用venv环境: codetalker_env"
    source codetalker_env/bin/activate
fi

# 内存配置
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

EXP_NAME="CodeTalker_s2"
DATASETS=("multidataset")
CONFIGS=("config/multidataset/stage2.yaml")

for idx in "${!DATASETS[@]}"; do
  DATASET_NAME="${DATASETS[$idx]}"
  CONFIG_FILE="${CONFIGS[$idx]}"

  if [ ! -f "${CONFIG_FILE}" ]; then
    echo "跳过 ${DATASET_NAME}: 找不到配置文件 ${CONFIG_FILE}"
    continue
  fi

  EXP_DIR="RUN/${DATASET_NAME}/${EXP_NAME}"
  MODEL_DIR="${EXP_DIR}/model"
  LOG_FILE="${EXP_DIR}/train-$(date +"%Y%m%d_%H%M%S").log"

  mkdir -p "${MODEL_DIR}" "${EXP_DIR}/result"

  STAGE1_MODEL="RUN/${DATASET_NAME}/CodeTalker_s1/model.pth.tar"
  if [ ! -f "${STAGE1_MODEL}" ]; then
      echo "错误: Stage1模型文件不存在: ${STAGE1_MODEL}，跳过 ${DATASET_NAME}"
      continue
  fi

  echo "=== Stage2训练配置 (${DATASET_NAME}) ==="
  echo "配置文件: ${CONFIG_FILE}"
  echo "Stage1模型: ${STAGE1_MODEL}"
  echo "输出目录: ${EXP_DIR}"
  echo "日志文件: ${LOG_FILE}"
  echo ""

  echo "验证Stage1模型..."
  python -c "
import torch
checkpoint = torch.load('${STAGE1_MODEL}', map_location='cpu')
print('✅ Stage1模型加载成功')
print('   - 包含的键:', list(checkpoint.keys())[:5])
if 'state_dict' in checkpoint:
    print('   - state_dict键数量:', len(checkpoint['state_dict']))
" || {
      echo "错误: Stage1模型文件损坏或格式不正确，跳过 ${DATASET_NAME}"
      continue
  }

  echo ""
  echo "开始Stage2训练 (${DATASET_NAME})..." | tee -a "${LOG_FILE}"
  nvidia-smi | tee -a "${LOG_FILE}"
  which python | tee -a "${LOG_FILE}"

  python -u main/train_pred.py \
    --config="${CONFIG_FILE}" \
    save_path "${EXP_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

  echo "Stage2训练完成 (${DATASET_NAME})。" | tee -a "${LOG_FILE}"
done
