import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import os
from collections import OrderedDict

def init_weights_(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d): # 确保包含 BatchNorm1d
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class VGG_A(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights_on_creation=True): # 修改了init_weights参数名以区分
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
        if init_weights_on_creation:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)


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

class VGG_A_Res_BN(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights_on_creation=True): # 修改了init_weights参数名
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock(inp_ch, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
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
        if init_weights_on_creation:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)

def evaluate_task2_model(checkpoint_path, model_type, device, batch_size=128, num_workers=2):
    print(f"开始评估模型: {checkpoint_path}")
    print(f"模型类型: {model_type}, 设备: {device}")

    # --- 1. 加载模型结构 ---
    if model_type == 'VGG_A':
        model = VGG_A(inp_ch=3, num_classes=10, init_weights_on_creation=False)
    elif model_type == 'VGG_A_Res_BN':
        model = VGG_A_Res_BN(inp_ch=3, num_classes=10, init_weights_on_creation=False)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}. 请选择 'VGG_A' 或 'VGG_A_Res_BN'.")

    # --- 2. 加载模型权重 ---
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点文件未找到 '{checkpoint_path}'")
        return float('nan'), float('nan')

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        

        state_dict_to_load = checkpoint
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict_to_load = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and not any(k.startswith('features.') or k.startswith('classifier.') for k in checkpoint.keys()):

            print("警告: 检查点似乎是一个字典但不是预期的格式。尝试直接加载。")


        new_state_dict = OrderedDict()
        for k, v in state_dict_to_load.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        print("模型权重加载成功。")
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        print("请确保模型类型与权重文件匹配，并且权重文件是有效的 state_dict。")
        return float('nan'), float('nan')

    model.to(device)
    model.eval()  # 设置模型为评估模式

    # --- 3. 准备 CIFAR-10 测试数据集 ---
    print("准备 CIFAR-10 测试数据集...")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda')
    except Exception as e:
        print(f"加载 CIFAR-10 数据集时出错: {e}")
        return float('nan'), float('nan')

    # --- 4. 评估模型 ---
    criterion = nn.CrossEntropyLoss() 
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

    avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
    accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0.
    
    print(f"评估完成: 平均损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%")
    return accuracy, avg_loss

def main():
    parser = argparse.ArgumentParser(description='Evaluate a Task2 VGG model on CIFAR-10 test set')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint file (.pth).')
    parser.add_argument('--model_type', type=str, required=True, choices=['VGG_A', 'VGG_A_Res_BN'],
                        help="Type of the model to evaluate: 'VGG_A' or 'VGG_A_Res_BN'.")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loader.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA evaluation.')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    evaluate_task2_model(args.checkpoint_path, args.model_type, device, args.batch_size, args.num_workers)

if __name__ == '__main__':
    main()