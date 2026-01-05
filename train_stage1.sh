#!/bin/bash
# Stage1 训练脚本：基于顶点的原始训练方式，适配 multidataset（JSON + 51维系数 -> 顶点）
set -e
export OMP_NUM_THREADS=10
export KMP_INIT_AT_FORK=FALSE
export PYTHONPATH=/home/caizhuoqiang/Code/audio_driven_baseline/CodeTalker:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

# 环境激活：优先 conda，其次 venv
if conda env list | grep -q "^codetalker "; then
    echo "使用conda环境: codetalker"
    conda activate codetalker
else
    echo "使用venv环境: codetalker_env"
    source codetalker_env/bin/activate
fi

DATASETS=("multidataset")
CONFIGS=("config/multidataset/stage1.yaml")
EXP_NAME="CodeTalker_s1"

for idx in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$idx]}"
  config="${CONFIGS[$idx]}"

  if [ ! -f "$config" ]; then
    echo "跳过 ${dataset}: 找不到配置文件 ${config}"
    continue
  fi

  exp_dir="RUN/${dataset}/${EXP_NAME}"
  model_dir="${exp_dir}/model"
  result_dir="${exp_dir}/result"
  now=$(date +"%Y%m%d_%H%M%S")
  log_file="${exp_dir}/train-${now}.log"

  mkdir -p "${model_dir}" "${result_dir}"

  echo "==== 开始训练 Stage1 | 数据集: ${dataset} | 配置: ${config} ====" | tee -a "${log_file}"
  echo $OMP_NUM_THREADS | tee -a "${log_file}"
  nvidia-smi | tee -a "${log_file}" 2>/dev/null || echo "No GPU available" | tee -a "${log_file}"

  python -u main/train_vq.py \
    --config="${config}" \
    save_path "${exp_dir}" \
    2>&1 | tee -a "${log_file}"
done
