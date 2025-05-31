import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# VGG-A模型（无BN）
class VGG_A(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# VGG-A模型（带BN）
class VGG_A_BN(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# 残差块定义
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# VGG-A变体：带残差块和BN
class VGG_A_Res_BN(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: 残差块
            ResidualBlock(inp_ch, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 2: 残差块
            ResidualBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 3: 两个残差块
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 4: 两个残差块
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 5: 两个残差块
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)


# 权重和方向处理函数
def get_weights(net):
    """获取模型所有参数的列表"""
    return [p.data for p in net.parameters()]

def set_weights(net, weights):
    """将权重列表设置回模型参数"""
    for p, w in zip(net.parameters(), weights):
        p.data.copy_(w)

def normalize_direction(direction, weights, norm='filter'):
    """规范化方向向量"""
    if norm == 'filter':
        for d, w in zip(direction, weights):
            if d.dim() > 1:
                d.mul_(w.norm() / (d.norm() + 1e-10))
    elif norm == 'layer':
        direction_norm = torch.cat([d.flatten() for d in direction]).norm()
        weights_norm = torch.cat([w.flatten() for w in weights]).norm()
        for d in direction:
            d.mul_(weights_norm / (direction_norm + 1e-10))
    return direction

def get_random_directions(net, device='cpu'):
    """生成两个随机的、规范化的方向向量"""
    weights = get_weights(net)
    direction1 = [torch.randn(w.size()).to(device) for w in weights]
    direction2 = [torch.randn(w.size()).to(device) for w in weights]
    direction1 = normalize_direction(direction1, weights, norm='filter')
    direction2 = normalize_direction(direction2, weights, norm='filter')
    return direction1, direction2

# 主函数
def main():
    parser = argparse.ArgumentParser(description='Visualize Loss Landscape of VGG-A Models')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, 
                        help='Paths to the trained model checkpoints (e.g., vgg_a_bn_adam_best.pth vgg_a_bn_sgd_best.pth vgg_a_no_bn_best.pth)')
    parser.add_argument('--x_min', type=float, default=-1.0, help='Min value for x-axis (alpha)')
    parser.add_argument('--x_max', type=float, default=1.0, help='Max value for x-axis (alpha)')
    parser.add_argument('--y_min', type=float, default=-1.0, help='Min value for y-axis (beta)')
    parser.add_argument('--y_max', type=float, default=1.0, help='Max value for y-axis (beta)')
    parser.add_argument('--num_steps', type=int, default=21, help='Number of steps for each axis (resolution)')
    parser.add_argument('--batch_size_viz', type=int, default=128, help='Batch size for calculating loss on the grid')
    parser.add_argument('--num_batches_viz', type=int, default=1, help='Number of batches to average loss over for each grid point')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the output plots')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--plot_3d', action='store_true', help='Generate a 3D surface plot instead of a 2D contour plot')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    print("Loading CIFAR10 data for loss calculation...")
    transform_viz = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 与训练时一致
    ])
    viz_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_viz)
    fixed_viz_batches = []
    temp_loader = torch.utils.data.DataLoader(viz_dataset, batch_size=args.batch_size_viz, shuffle=False)
    for i, data in enumerate(temp_loader):
        if i < args.num_batches_viz:
            fixed_viz_batches.append(data)
        else:
            break
    if not fixed_viz_batches:
        print("错误：未能从数据加载器中获取任何批次用于可视化。请检查 batch_size_viz 和数据集。")
        return
    print(f"Using {len(fixed_viz_batches)} batch(es) of size {args.batch_size_viz} for loss evaluation.")

    # 模型和权重文件映射
    model_configs = [
        ('BN_Adam', 'vgg_a_res_bn_adam_best.pth', VGG_A_Res_BN),
        ('BN_SGD', 'vgg_a_res_bn_sgd_best.pth', VGG_A_Res_BN),
        ('No_BN_SGD', 'vgg_a_no_bn_best.pth', VGG_A)
    ]

    # 计算每个模型的损失景观
    for model_name, model_path, model_class in model_configs:
        if model_path not in args.model_paths:
            print(f"跳过 {model_name}，因为未提供 {model_path}")
            continue

        print(f"\nProcessing model: {model_name} ({model_path})")
        if not os.path.exists(model_path):
            print(f"错误: 模型检查点文件未找到 '{model_path}'")
            continue

        # 初始化模型
        model = model_class(inp_ch=3, num_classes=10, init_weights=False).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        # 处理可能的 'module.' 前缀
        state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        try:
            model.load_state_dict(new_state_dict)
        except RuntimeError as e:
            print(f"错误: 加载 {model_path} 时模型结构不匹配: {e}")
            continue
        model.eval()
        print(f"Model {model_name} loaded successfully.")

        # 获取初始权重和随机方向
        initial_weights = get_weights(model)
        direction1, direction2 = get_random_directions(model, device=device)

        # 创建坐标网格
        x_coords = np.linspace(args.x_min, args.x_max, args.num_steps)
        y_coords = np.linspace(args.y_min, args.y_max, args.num_steps)
        loss_grid = np.zeros((args.num_steps, args.num_steps))

        # 计算损失网格
        criterion = nn.CrossEntropyLoss()
        model_copy = copy.deepcopy(model)
        print(f"Calculating loss grid for {model_name}...")
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                new_weights = [w_init + x * d1_val + y * d2_val for w_init, d1_val, d2_val in zip(initial_weights, direction1, direction2)]
                set_weights(model_copy, new_weights)
                model_copy.eval()

                current_grid_loss = 0.0
                num_samples_processed = 0
                with torch.no_grad():
                    for inputs, targets in fixed_viz_batches:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model_copy(inputs)
                        loss = criterion(outputs, targets)
                        current_grid_loss += loss.item() * inputs.size(0)
                        num_samples_processed += inputs.size(0)
                loss_grid[j, i] = current_grid_loss / num_samples_processed
            print(f"Progress: {(i+1) / args.num_steps * 100:.1f}% completed ({i+1}/{args.num_steps} x-coordinates)")

        # 绘制损失景观
        print(f"Plotting loss landscape for {model_name}...")
        fig = plt.figure(figsize=(12, 10) if args.plot_3d else (10, 8))
        X, Y = np.meshgrid(x_coords, y_coords)
        if args.plot_3d:
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, loss_grid, cmap=cm.viridis, linewidth=0, antialiased=False)
            ax.set_xlabel(r'$\alpha$ (Direction 1)')
            ax.set_ylabel(r'$\beta$ (Direction 2)')
            ax.set_zlabel('Loss')
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.title(f'3D Loss Landscape\nModel: {model_name}')
        else:
            contour = plt.contourf(X, Y, loss_grid, levels=20, cmap=cm.viridis)
            plt.colorbar(contour, label='Loss')
            plt.xlabel(r'$\alpha$ (Direction 1)')
            plt.ylabel(r'$\beta$ (Direction 2)')
            plt.title(f'2D Loss Landscape Contour\nModel: {model_name}')
            plt.minorticks_on()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        output_path = os.path.join(args.output_dir, f'loss_landscape_{model_name.lower()}.png')
        plt.savefig(output_path, dpi=300)
        print(f"Loss landscape plot saved to {output_path}")
        plt.close()

if __name__ == '__main__':
    main()