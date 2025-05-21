import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import json                     # 用于保存结构化的结果
import matplotlib.pyplot as plt # 用于绘图

# 从之前的代码文件中导入必要的模块
from data.loaders import get_cifar_loader
from models.vgg import VGG_A
# from models.vgg import VGG_A_BatchNorm # 如果你要测试包含BN的模型，取消注释
# from models.vgg import VGG_A_Dropout  # 如果你要测试包含Dropout的模型，取消注释
# from utils.nn import init_weights_ # VGG_A 内部已使用

# =============================================================================
# 0. 配置参数、路径和实验组合
# =============================================================================
# --- 基本配置 ---
BATCH_SIZE = 128
EPOCHS = 50           # 增加训练轮数以更好收敛
DATA_DIR = './data/'  # CIFAR-10 数据存储/下载路径
NUM_WORKERS = 4
SEED = 42             # 设置随机种子以便结果可复现

# --- 结果和图像保存路径 ---
RESULTS_DIR = "/remote-home/yinzhitao/pj2/codes/VGG_BatchNorm/task1_train_res"
FIGURES_DIR = "/remote-home/yinzhitao/pj2/codes/VGG_BatchNorm/task1_train_fig"
MODEL_SAVE_DIR = os.path.join(RESULTS_DIR, "models") # 模型权重保存在结果目录的子文件夹中

# --- 定义要测试的实验组合 ---
# 要求尝试不同损失函数
# 要求尝试不同优化器 (或自己实现 )
EXPERIMENT_CONFIGS = [
    # 组合 1: CrossEntropy + Adam (默认学习率)
    {"loss_fn": nn.CrossEntropyLoss, "loss_name": "CrossEntropy",
     "optimizer": optim.Adam, "opt_name": "Adam",
     "opt_params": {"lr": 0.001}},

    # 组合 2: CrossEntropy + SGD (常用参数)
    {"loss_fn": nn.CrossEntropyLoss, "loss_name": "CrossEntropy",
     "optimizer": optim.SGD, "opt_name": "SGD",
     "opt_params": {"lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4}},

    # 组合 3: CrossEntropy + Adam (不同学习率)
    {"loss_fn": nn.CrossEntropyLoss, "loss_name": "CrossEntropy",
     "optimizer": optim.Adam, "opt_name": "Adam",
     "opt_params": {"lr": 0.0005}},

    # 组合 4: CrossEntropy + SGD (不同学习率)
    {"loss_fn": nn.CrossEntropyLoss, "loss_name": "CrossEntropy",
     "optimizer": optim.SGD, "opt_name": "SGD",
     "opt_params": {"lr": 0.005, "momentum": 0.9, "weight_decay": 5e-4}},

    # --- 在这里添加更多你想测试的组合 ---
    # 例如：不同的优化器 RMSprop
    # {"loss_fn": nn.CrossEntropyLoss, "loss_name": "CrossEntropy",
    #  "optimizer": optim.RMSprop, "opt_name": "RMSprop",
    #  "opt_params": {"lr": 0.001}},
    # 例如：不同的损失函数 (需要模型最后输出 log_softmax)
    # {"loss_fn": nn.NLLLoss, "loss_name": "NLLLoss", ... }
    # 注意: 如果使用 NLLLoss，模型 forward 最后需要加 F.log_softmax(x, dim=1)
]

# 设置随机种子
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
# 可能需要设置 deterministic 以获得完全可复现性（但可能影响性能）
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


# =============================================================================
# 1. 辅助函数：绘图并保存
# =============================================================================
def plot_and_save_curves(train_losses, val_losses, val_accuracies, run_name, save_dir):
    """
    绘制训练/验证损失和验证准确率曲线，并保存图像。

    输入参数:
        train_losses (list): 每轮的训练损失列表。
        val_losses (list): 每轮的验证损失列表。
        val_accuracies (list): 每轮的验证准确率列表。
        run_name (str): 当前实验运行的名称，用于文件名。
        save_dir (str): 保存图像的目录路径。
    """
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{run_name} - Loss Curves')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{run_name} - Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{run_name}_curves.png")
    plt.savefig(save_path)
    plt.close() # 关闭图像，防止在循环中显示或占用过多内存

# =============================================================================
# 2. 修改后的训练函数
# =============================================================================
def train_model(model, trainloader, valloader, criterion, optimizer, device, epochs, model_save_path, run_name):
    """
    训练模型，记录历史，并保存最佳权重。

    输入参数:
        ... (同前) ...
        model_save_path (str): 保存此运行最佳模型权重的路径。
        run_name (str): 用于打印日志的运行名称。

    输出参数:
        train_losses, val_losses, val_accuracies (list): 训练历史记录。
        best_run_val_accuracy (float): 此运行中达到的最佳验证准确率。
    """
    model.to(device)
    best_run_val_accuracy = 0.0
    train_losses, val_losses, val_accuracies = [], [], []

    print(f"[{run_name}] Starting training for {epochs} epochs...")
    run_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        samples_processed = 0

        # 使用 tqdm 显示训练进度条 (可选)
        # from tqdm import tqdm
        # train_iterator = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} Training")
        # for data in train_iterator:

        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            samples_processed += inputs.size(0)

            # 可选: 更新 tqdm 进度条描述
            # if i % 50 == 0:
            #     train_iterator.set_postfix({"Loss": loss.item()})


        epoch_train_loss = running_loss / samples_processed
        train_losses.append(epoch_train_loss)

        # 在验证集上评估
        epoch_val_loss, epoch_val_accuracy = evaluate_model(model, valloader, criterion, device)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        epoch_duration = time.time() - epoch_start_time

        print(f"[{run_name}] Epoch [{epoch + 1}/{epochs}] - {epoch_duration:.2f}s - "
              f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}%")

        # 保存此运行中验证准确率最高的模型
        if epoch_val_accuracy > best_run_val_accuracy:
            best_run_val_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(), model_save_path)
            # print(f"    -> New best model for this run saved (Val Acc: {best_run_val_accuracy:.2f}%)") # 可以取消注释以显示保存信息

    run_duration = time.time() - run_start_time
    print(f"[{run_name}] Finished Training in {run_duration:.2f} seconds. Best Val Acc for this run: {best_run_val_accuracy:.2f}%")

    return train_losses, val_losses, val_accuracies, best_run_val_accuracy

# =============================================================================
# 3. 评估函数 (保持不变)
# =============================================================================
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = 100.0 * correct_predictions / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy

# =============================================================================
# 4. 主程序入口
# =============================================================================
if __name__ == '__main__':
    # 检查 CUDA 可用性 (如果仍然有问题，需要先解决)
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. Please check your installation and environment.")
        # 你可以选择退出，或者强制使用 CPU (但不推荐用于训练 VGG)
        # device = torch.device("cpu")
        # print("Warning: CUDA not found, running on CPU (slow).")
        exit() # 强制退出如果 CUDA 不可用
    else:
        device = torch.device("cuda")
        print(f"Using device: {device}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


    # 创建结果和图像目录
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        print(f"Results will be saved in: {RESULTS_DIR}")
        print(f"Figures will be saved in: {FIGURES_DIR}")
        print(f"Models will be saved in: {MODEL_SAVE_DIR}")
    except OSError as e:
        print(f"Error creating directories: {e}")
        exit() # 如果无法创建目录则退出


    # 加载数据
    print("\nLoading CIFAR-10 data...")
    try:
        train_loader = get_cifar_loader(root=DATA_DIR, batch_size=BATCH_SIZE, train=True, shuffle=True, num_workers=NUM_WORKERS)
        test_loader = get_cifar_loader(root=DATA_DIR, batch_size=BATCH_SIZE, train=False, shuffle=False, num_workers=NUM_WORKERS)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit() # 如果数据加载失败则退出

    # --- 存储所有实验的最佳结果 ---
    all_run_summaries = []

    # --- 开始循环运行实验 ---
    print(f"\nStarting {len(EXPERIMENT_CONFIGS)} experiments...")
    for i, config in enumerate(EXPERIMENT_CONFIGS):
        run_log_prefix = f"[Run {i+1}/{len(EXPERIMENT_CONFIGS)}]"
        print(f"\n{run_log_prefix} --- Configuration ---")
        print(f"  Loss Function: {config['loss_name']}")
        print(f"  Optimizer: {config['opt_name']}")
        print(f"  Optimizer Params: {config['opt_params']}")

        # 1. 构造运行名称和保存路径 (更详细的名称)
        opt_params_str = "_".join([f"{k}{v}" for k, v in config['opt_params'].items()])
        run_name = f"VGGA_{config['loss_name']}_{config['opt_name']}_{opt_params_str}"
        model_save_path = os.path.join(MODEL_SAVE_DIR, f"{run_name}_best_ep{EPOCHS}.pth")
        figure_save_path = os.path.join(FIGURES_DIR, f"{run_name}_curves_ep{EPOCHS}.png")

        # 2. 重新初始化模型 (确保每次实验权重独立)
        #    可以根据需要选择不同的模型，例如 VGG_A_BatchNorm
        model = VGG_A(num_classes=len(classes), init_weights=True)
        model.to(device) # 移动到设备
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model: VGG_A ({num_params:,} params)")


        # 3. 实例化损失函数和优化器
        criterion = config["loss_fn"]()
        try:
            optimizer = config["optimizer"](model.parameters(), **config["opt_params"])
        except Exception as e:
            print(f"Error initializing optimizer {config['opt_name']} with params {config['opt_params']}: {e}")
            continue # 跳过这个错误的配置

        # 4. 训练模型并获取历史记录和该运行的最佳准确率
        try:
            history = train_model(model, train_loader, test_loader, criterion, optimizer, device, EPOCHS, model_save_path, run_name)
            train_losses, val_losses, val_accuracies, run_best_val_acc = history
        except Exception as e:
            print(f"Error during training for run {run_name}: {e}")
            continue # 训练出错则跳过后续处理

        # 5. 找到最佳准确率对应的轮数
        if val_accuracies: # 确保列表不为空
             # 使用 run_best_val_acc 替代 max(val_accuracies) 以确保一致性
            best_epoch_index = val_accuracies.index(run_best_val_acc)
            best_epoch_num = best_epoch_index + 1 # 转换为 1-based 轮数
        else:
            run_best_val_acc = -1.0
            best_epoch_num = -1


        # 6. 记录本次运行的总结信息
        run_summary = {
            "run_name": run_name,
            "config_index": i+1,
            "loss_function": config['loss_name'],
            "optimizer": config['opt_name'],
            "optimizer_params": config['opt_params'],
            "epochs_trained": EPOCHS,
            "best_validation_accuracy": run_best_val_acc,
            "best_epoch": best_epoch_num,
            "saved_model_path": model_save_path if run_best_val_acc > 0 else None,
            "figure_path": figure_save_path
        }
        all_run_summaries.append(run_summary)

        print(f"{run_log_prefix} --- Summary ---")
        print(f"  Best Validation Accuracy: {run_best_val_acc:.2f}% at Epoch {best_epoch_num}")
        if run_best_val_acc > 0:
             print(f"  Best model weights saved to: {model_save_path}")

        # 7. 绘制并保存训练曲线图
        try:
            plot_and_save_curves(train_losses, val_losses, val_accuracies, run_name, FIGURES_DIR)
            print(f"  Training curves plot saved to: {figure_save_path}")
        except Exception as e:
             print(f"Error plotting/saving curves for run {run_name}: {e}")


    # --- 所有实验结束后，保存总体结果总结 ---
    summary_file_path = os.path.join(RESULTS_DIR, f"all_experiments_summary_ep{EPOCHS}.json")
    print(f"\n--- All Experiments Finished ---")
    try:
        with open(summary_file_path, 'w') as f:
            json.dump(all_run_summaries, f, indent=4)
        print(f"Overall summary of all runs saved to: {summary_file_path}")
    except Exception as e:
        print(f"Error saving overall summary file: {e}")

    # (可选) 打印一个简单的排序结果
    if all_run_summaries:
        all_run_summaries.sort(key=lambda x: x['best_validation_accuracy'], reverse=True)
        print("\n--- Top Performing Configurations ---")
        for rank, summary in enumerate(all_run_summaries[:5]): # 打印前 5 名
             print(f"  Rank {rank+1}: {summary['run_name']} - Best Val Acc: {summary['best_validation_accuracy']:.2f}% (Epoch {summary['best_epoch']})")