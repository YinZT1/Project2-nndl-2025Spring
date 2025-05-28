import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import os
from collections import OrderedDict


from models.vgg16 import VGG


def evaluate_model(model, dataloader, device, criterion=None):
    """评估模型在给定dataloader上的损失和准确率"""
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # 在评估时不需要计算梯度
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if criterion:
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

    avg_loss = total_loss / total_samples if criterion and total_samples > 0 else float('nan')
    accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0.
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluate a VGG model checkpoint on CIFAR10 train and test sets')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint file (.pth.tar)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loader')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA evaluation')
    args = parser.parse_args()

    # --- 设置设备 ---
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # --- 加载模型检查点 ---
    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 检查点文件未找到 '{args.checkpoint_path}'")
        return

    print(f"加载检查点: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # 从检查点中获取模型配置参数 (如果存在)
    # 这些参数是在训练脚本中保存的 'args'
    model_args = checkpoint.get('args', None)

    vgg_config_name = getattr(model_args, 'vgg_config', 'VGG16_standard') # 提供默认值以防万一
    activation_name = getattr(model_args, 'activation', 'relu')
    num_classes = 10 # CIFAR10

    print(f"模型配置: VGG Config='{vgg_config_name}', Activation='{activation_name}'")
    use_log_softmax_for_model = (getattr(model_args, 'loss_fn', 'cross_entropy') == 'nll_loss')

    model = VGG(vgg_name=vgg_config_name,
                activation_name=activation_name,
                num_classes=num_classes,
                use_log_softmax=use_log_softmax_for_model, # 保持与训练时一致
                init_weights=False)  # 我们将加载预训练权重

    # 加载模型状态字典
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # 移除 'module.' 前缀 (如果模型是用DataParallel保存的)
        new_state_dict[name] = v
    
    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"加载 state_dict 时出错: {e}")
        print("这可能是由于模型定义与检查点中的权重不匹配。")
        print("请确保用于评估的模型结构与训练时完全一致。")
        return
        
    model.to(device)
    print("模型已成功加载并移至设备。")

    # --- 准备 CIFAR10 数据集 ---
    print("准备 CIFAR10 数据集...")
    # 标准的 CIFAR10 变换 (与训练脚本中的测试集变换一致)
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    try:
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_eval)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_cuda)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_eval)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_cuda)
    except Exception as e:
        print(f"加载CIFAR10数据集时出错: {e}")
        print("请确保您可以访问网络以下载数据集，或者数据集已存在于 './data' 目录。")
        return

    # --- 定义损失函数（可选，主要用于报告损失） ---
    criterion = nn.CrossEntropyLoss() # 假设主要评估指标是准确率

    # --- 在训练集上评估 ---
    print("\n在训练集上评估...")
    train_loss, train_acc = evaluate_model(model, trainloader, device, criterion)
    print(f"训练集: 平均损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")

    # --- 在测试集上评估 ---
    print("\n在测试集上评估...")
    test_loss, test_acc = evaluate_model(model, testloader, device, criterion)
    print(f"测试集: 平均损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%")

    print("\n评估完成。")

if __name__ == '__main__':
    main()