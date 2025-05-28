#!/bin/bash

# --- 基本配置 ---
EPOCHS=100                 # 每个模型的训练轮数 (您可以先用较小值测试, e.g., 10)
BATCH_SIZE=128
NUM_WORKERS=2
BASE_CHECKPOINT_DIR="./grid_search_results" # 所有实验结果的根目录
PYTHON_COMMAND="python -m newvgg16" # 您运行训练的命令

# --- 参数组合 ---
VGG_CONFIGS=("VGG16_standard" "VGG16_light")
ACTIVATIONS=("relu" "leaky_relu" "tanh")
LOSS_FNS=("cross_entropy" "nll_loss")
OPTIMIZERS=("adam" "sgd")

# --- 优化器特定参数 ---
LR_ADAM="0.0001"
LR_SGD="0.01"           # SGD 通常需要较高的初始学习率
MOMENTUM_SGD="0.9"
WEIGHT_DECAY_SGD="1e-4" # SGD 和 AdamW 会使用此参数 (Adam 则忽略)
WEIGHT_DECAY_ADAMW="1e-4" # 如果将来使用 adamw，可以参考

# 确保基础检查点目录存在
mkdir -p $BASE_CHECKPOINT_DIR

# 实验计数器
experiment_count=0

echo "Starting Grid Search for 24 VGG configurations..."
echo "=================================================="
echo "Base Python command: $PYTHON_COMMAND"
echo "Epochs per run: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Results will be saved in: $BASE_CHECKPOINT_DIR"
echo "=================================================="

# --- 循环遍历所有配置 ---
for vgg_c in "${VGG_CONFIGS[@]}"; do
  for act_f in "${ACTIVATIONS[@]}"; do
    for loss_f in "${LOSS_FNS[@]}"; do
      for opt_f in "${OPTIMIZERS[@]}"; do

        experiment_count=$((experiment_count + 1))
        run_name="run_${experiment_count}_${vgg_c}_${act_f}_${loss_f}_${opt_f}"
        checkpoint_dir="${BASE_CHECKPOINT_DIR}/${run_name}"
        mkdir -p "$checkpoint_dir" # 为当前实验创建单独的检查点目录

        echo ""
        echo "--- Starting Experiment #$experiment_count ---"
        echo "Configuration:"
        echo "  VGG Config:   $vgg_c"
        echo "  Activation:   $act_f"
        echo "  Loss Function:$loss_f"
        echo "  Optimizer:    $opt_f"
        echo "  Run Name:     $run_name"
        echo "  Checkpoint Dir: $checkpoint_dir"

        # 构建命令行参数
        cmd_args=""
        cmd_args+=" --vgg_config $vgg_c"
        cmd_args+=" --activation $act_f"
        cmd_args+=" --loss_fn $loss_f"
        cmd_args+=" --optimizer $opt_f"
        cmd_args+=" --epochs $EPOCHS"
        cmd_args+=" --batch_size $BATCH_SIZE"
        cmd_args+=" --num_workers $NUM_WORKERS"
        cmd_args+=" --run_name \"$run_name\"" # 加引号以防参数中有特殊字符 (虽然这里没有)
        cmd_args+=" --checkpoint_dir \"$checkpoint_dir\""

        # 根据优化器设置特定参数
        if [ "$opt_f" == "sgd" ]; then
          cmd_args+=" --lr $LR_SGD"
          cmd_args+=" --momentum $MOMENTUM_SGD"
          cmd_args+=" --weight_decay $WEIGHT_DECAY_SGD"
          echo "  SGD Params:   LR=$LR_SGD, Momentum=$MOMENTUM_SGD, WD=$WEIGHT_DECAY_SGD"
        elif [ "$opt_f" == "adam" ]; then
          cmd_args+=" --lr $LR_ADAM"
          # Adam 优化器通常不直接在构造函数中接收 weight_decay,
          # 脚本 task1_train_vgg16.py 中 optim.Adam 未使用 weight_decay
          # 如果需要 Adam 的权重衰减, 应使用 adamw 优化器
          echo "  Adam Params:  LR=$LR_ADAM"
        fi
        
        # (可选) 如果您想为 adamw 单独设置参数:
        # elif [ "$opt_f" == "adamw" ]; then
        #   cmd_args+=" --lr $LR_ADAM" # AdamW 通常也用类似 Adam 的 LR
        #   cmd_args+=" --weight_decay $WEIGHT_DECAY_ADAMW"
        #   echo "  AdamW Params: LR=$LR_ADAM, WD=$WEIGHT_DECAY_ADAMW"
        # fi

        full_command="$PYTHON_COMMAND $cmd_args"
        echo "Executing: $full_command"

        # 执行训练命令，并将标准输出和标准错误重定向到日志文件
        # 如果您希望在终端实时看到输出，可以移除 `> >(tee ...)` 部分，或只用 `| tee ...`
        # $full_command > "${checkpoint_dir}/train_log.txt" 2>&1
        # 为了能在终端看到进度，同时记录日志:
        eval $full_command | tee "${checkpoint_dir}/train_log.txt"
        
        # 检查命令是否成功执行 (可选)
        if [ $? -eq 0 ]; then
          echo "Experiment #$experiment_count ($run_name) COMPLETED successfully."
        else
          echo "Experiment #$experiment_count ($run_name) FAILED. Check log: ${checkpoint_dir}/train_log.txt"
        fi
        echo "--------------------------------------"

      done # optimizers
    done   # loss_fns
  done     # activations
done       # vgg_configs

echo ""
echo "=================================================="
echo "All $experiment_count experiments have been launched."
echo "Results and logs are in subdirectories under: $BASE_CHECKPOINT_DIR"
echo "=================================================="