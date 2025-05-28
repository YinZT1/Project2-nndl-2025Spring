# VGG_Loss_Landscape.py (完善和注释建议)
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import copy # 用于深拷贝模型
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # 用于更丰富的颜色映射
from mpl_toolkits.mplot3d import Axes3D # 用于3D绘图 (如果需要)

# 假设您的VGG模型定义在这里，或者可以正确导入
# from models.vgg16 import VGG # 确保这个导入路径正确
# (根据您之前提供的 newvgg16.py，这个导入是正确的，前提是项目结构支持)
# 如果运行时提示找不到 models.vgg16，请检查您的PYTHONPATH或运行脚本的当前目录
try:
    from models.vgg16 import VGG
except ModuleNotFoundError:
    print("错误: 无法导入 'from models.vgg16 import VGG'.")
    print("请确保 VGG_Loss_Landscape.py 与 models 文件夹在正确的相对路径，")
    print("或者您的项目已正确设置为Python包。")
    print("如果您将 VGG_Loss_Landscape.py 放在 task1 目录下，并且 models 也在 task1 下，")
    print("那么导入应该是 'from models.vgg16 import VGG'")
    print("如果 VGG_Loss_Landscape.py 和 models 文件夹都在项目的根目录下，那也是对的。")
    exit()


# --- 权重和方向处理函数 (来自您提供的脚本，稍作整理和注释) ---
def get_weights(net):
    """ 获取模型所有参数的列表 """
    return [p.data for p in net.parameters()]

def set_weights(net, weights):
    """ 将权重列表设置回模型参数 """
    for p, w in zip(net.parameters(), weights):
        p.data.copy_(w)

def normalize_direction(direction, weights, norm='filter'):
    """
    规范化方向向量。
    Args:
        direction: 一个与 weights 结构相同的方向向量列表。
        weights: 模型的原始权重列表。
        norm: 规范化类型 ('filter' 或 'layer')。
    """
    if norm == 'filter':
        # 按每个滤波器的范数进行规范化
        for d, w in zip(direction, weights):
            # d.dim() > 1 确保我们只处理权重张量 (例如 Conv, Linear)，而不是标量 (例如某些BN参数，尽管我们通常不扰动BN统计数据)
            if d.dim() > 1: # 通常权重张量至少是2D (out_features, in_features) 或 (out_c, in_c, kH, kW)
                # 对于卷积层 (out_c, in_c, kH, kW), d.norm(dim=(1,2,3)...) 可能不直接适用
                # Li et al. 的论文中做法是 d_i = d_i * ||w_i|| / ||d_i|| (Frobenius范数)
                # 这里的实现是 d_i = d_i / ||d_i|| * ||w_i||
                # 让我们确保对每个“filter”或“neuron”的权重块进行操作
                if d.dim() == 4: # Conv2d: (out_channels, in_channels, H, W)
                    # 对每个 (in_channels, H, W) 块进行规范化是困难的
                    # 通常是整个权重张量 (例如一个Conv2d的weight) 进行规范化
                    d.mul_(w.norm() / (d.norm() + 1e-10)) # 1e-10 防止除以0
                elif d.dim() == 2: # Linear: (out_features, in_features)
                    d.mul_(w.norm() / (d.norm() + 1e-10))
                # 其他情况可以根据需要添加
    elif norm == 'layer':
        # 按每层的范数进行规范化 (这里简化为整个方向向量的范数)
        direction_norm = torch.cat([d.flatten() for d in direction]).norm()
        weights_norm = torch.cat([w.flatten() for w in weights]).norm()
        for d in direction:
            d.mul_(weights_norm / (direction_norm + 1e-10))
    # 也可以不进行与weights相关的规范化，只将方向向量本身规范化为单位向量
    # 例如：
    # for d in direction:
    #     d.div_(d.norm() + 1e-10)
    return direction


def get_random_directions(net, device='cpu'): # 您脚本中函数名是 get_random_ δύο_directions
    """ 生成两个随机的、规范化的方向向量 """
    weights = get_weights(net) # 获取当前模型的权重作为参考
    direction1 = [torch.randn(w.size()).to(device) for w in weights] # 第一个随机方向
    direction2 = [torch.randn(w.size()).to(device) for w in weights] # 第二个随机方向

    # 规范化方向 (Li et al. 2018 的 filter-wise normalization)
    # 这一步很重要，确保扰动与原始权重的大小成比例
    direction1 = normalize_direction(direction1, weights, norm='filter')
    direction2 = normalize_direction(direction2, weights, norm='filter')

    return direction1, direction2


# --- 主函数 ---
def main():
    parser = argparse.ArgumentParser(description='Visualize Loss Landscape of a VGG Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pth.tar or .pt)')
    parser.add_argument('--vgg_config', type=str, default='VGG16_standard', help='VGG configuration name (e.g., VGG16_standard)')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function used in the model')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes (e.g., 10 for CIFAR10)')

    parser.add_argument('--x_min', type=float, default=-1.0, help='Min value for x-axis (alpha)')
    parser.add_argument('--x_max', type=float, default=1.0, help='Max value for x-axis (alpha)')
    parser.add_argument('--y_min', type=float, default=-1.0, help='Min value for y-axis (beta)')
    parser.add_argument('--y_max', type=float, default=1.0, help='Max value for y-axis (beta)')
    parser.add_argument('--num_steps', type=int, default=21, help='Number of steps for each axis (resolution)') # 默认21x21的网格

    parser.add_argument('--batch_size_viz', type=int, default=128, help='Batch size for calculating loss on the grid')
    parser.add_argument('--num_batches_viz', type=int, default=1, help='Number of batches to average loss over for each grid point (1 means use one batch)')

    parser.add_argument('--output_path', type=str, default='loss_landscape.png', help='Path to save the output plot')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--plot_3d', action='store_true', help='Generate a 3D surface plot instead of a 2D contour plot')

    args = parser.parse_args()

    # --- 设置设备 ---
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA for computations.")
    else:
        device = torch.device('cpu')
        print("Using CPU for computations.")

    # --- 加载数据 ---
    print("Loading CIFAR10 data for loss calculation...")
    # 使用与训练时相似的变换，但不包括随机数据增强
    transform_viz = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    # 使用测试集或训练集的一个子集来计算损失
    viz_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_viz)
    # 注意：如果数据集很大，并且 num_batches_viz > 1，这里可能需要一个子集
    # viz_loader = torch.utils.data.DataLoader(viz_dataset, batch_size=args.batch_size_viz, shuffle=False)
    # 为了快速测试，可以只用少量数据；为了更准确，可以用整个验证集（如果 num_batches_viz 设为足够大）
    # 为了确保每次运行用相同数据评估损失，可以不打乱shuffle=False
    # 如果 num_batches_viz 较大，shuffle=True 可能更好以获得更平均的损失
    # 但通常为了可复现性，可视化时用 shuffle=False 且固定几批数据
    
    # 创建一个迭代器，以便可以获取固定数量的批次
    fixed_viz_batches = []
    temp_loader = torch.utils.data.DataLoader(viz_dataset, batch_size=args.batch_size_viz, shuffle=False) # Shuffle False for consistency
    for i, data in enumerate(temp_loader):
        if i < args.num_batches_viz:
            fixed_viz_batches.append(data)
        else:
            break
    if not fixed_viz_batches:
        print("错误：未能从数据加载器中获取任何批次用于可视化。请检查 batch_size_viz 和数据集。")
        return
    print(f"Using {len(fixed_viz_batches)} batch(es) of size {args.batch_size_viz} for loss evaluation at each grid point.")


    # --- 加载预训练模型 ---
    print(f"Loading model: {args.model_path}")
    # 假设模型是在 newvgg16.py 中训练的，num_classes=10
    # use_log_softmax 在这里通常设为 False，因为我们会直接用 CrossEntropyLoss
    model = VGG(vgg_name=args.vgg_config, activation_name=args.activation, num_classes=args.num_classes, init_weights=False, use_log_softmax=False)

    if not os.path.exists(args.model_path):
        print(f"错误: 模型检查点文件未找到 '{args.model_path}'")
        return

    checkpoint = torch.load(args.model_path, map_location=device)
    # 处理可能的 'module.' 前缀 (如果模型是用 DataParallel 保存的)
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval() # 非常重要：设置为评估模式
    print("Model loaded successfully.")

    # 保存原始权重
    initial_weights = get_weights(model)

    # --- 生成随机方向 ---
    print("Generating random directions...")
    direction1, direction2 = get_random_directions(model, device=device)
    # （可选）可以保存方向向量以便复现
    # torch.save({'direction1': direction1, 'direction2': direction2}, 'random_directions.pt')

    # --- 创建坐标网格 ---
    x_coords = np.linspace(args.x_min, args.x_max, args.num_steps)
    y_coords = np.linspace(args.y_min, args.y_max, args.num_steps)
    loss_grid = np.zeros((args.num_steps, args.num_steps))

    # --- 计算损失网格上的损失值 ---
    criterion = nn.CrossEntropyLoss() # 通常使用交叉熵损失
    model_copy = copy.deepcopy(model) # 创建一个模型副本进行权重修改和评估

    print("Calculating loss on the grid...")
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            # 构造新的权重: w = w_initial + x * d1 + y * d2
            new_weights = []
            for w_init, d1_val, d2_val in zip(initial_weights, direction1, direction2):
                new_weights.append(w_init + x * d1_val + y * d2_val)
            
            set_weights(model_copy, new_weights) # 将新权重设置到模型副本
            model_copy.eval() # 确保副本也是评估模式

            current_grid_loss = 0.0
            num_samples_processed = 0
            with torch.no_grad(): # 在评估损失时不需要计算梯度
                for inputs, targets in fixed_viz_batches: # 使用固定的批次数据
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model_copy(inputs)
                    loss = criterion(outputs, targets)
                    current_grid_loss += loss.item() * inputs.size(0) # 乘以batch size得到总损失
                    num_samples_processed += inputs.size(0)
            
            loss_grid[j, i] = current_grid_loss / num_samples_processed # 存储平均损失 (j对应y, i对应x)
            
        print(f"Progress: { (i+1) / args.num_steps * 100 :.1f}% completed ({i+1}/{args.num_steps} x-coordinates)")

    # --- 绘图 ---
    print("Plotting loss landscape...")
    fig = plt.figure(figsize=(10, 8) if not args.plot_3d else (12,10)) # 根据2D或3D调整大小
    X, Y = np.meshgrid(x_coords, y_coords)

    if args.plot_3d:
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, loss_grid, cmap=cm.viridis, # 使用 'viridis' 或其他漂亮的 colormap
                               linewidth=0, antialiased=False)
        ax.set_xlabel(r'$\alpha$ (Direction 1)')
        ax.set_ylabel(r'$\beta$ (Direction 2)')
        ax.set_zlabel('Loss')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title(f'3D Loss Landscape\nModel: {os.path.basename(args.model_path)}\nConfig: {args.vgg_config}, Activation: {args.activation}')
    else: # 2D contour plot
        # 可以调整 levels 的数量或直接让 contourf 自动选择
        # levels = np.linspace(loss_grid.min(), loss_grid.min() + (loss_grid.max() - loss_grid.min()) * 0.5, 20) # 关注损失较低的区域
        contour = plt.contourf(X, Y, loss_grid, levels=20, cmap=cm.viridis) # levels可以调整等高线条数
        plt.colorbar(contour, label='Loss')
        # plt.contour(X, Y, loss_grid, levels=contour.levels, colors='k', linewidths=0.5) # 添加黑色轮廓线
        plt.xlabel(r'$\alpha$ (Direction 1)')
        plt.ylabel(r'$\beta$ (Direction 2)')
        plt.title(f'2D Loss Landscape Contour\nModel: {os.path.basename(args.model_path)}\nConfig: {args.vgg_config}, Activation: {args.activation}')
        plt.minorticks_on() # 显示次刻度线
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)


    plt.savefig(args.output_path, dpi=300) # 保存图像，可以调整dpi
    print(f"Loss landscape plot saved to {args.output_path}")
    plt.show()


if __name__ == '__main__':
    main()