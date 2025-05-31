import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

# 设置随机种子以确保可复现性
torch.manual_seed(42)
np.random.seed(42)

# 获取模型参数数量
def get_number_of_parameters(model):
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()
    return parameters_n

# 初始化权重
def init_weights_(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
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

# VGG-A原始模型（无BN，来自vgg.py）
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
            init_weights_(m)

# 训练函数
def train_model(model, train_loader, test_loader, optimizer, criterion, device, epochs=50, model_name="model"):
    train_losses = []
    test_accuracies = []
    best_acc = 0.0
    best_model_path = f"{model_name}_best.pth"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # 测试模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), best_model_path)

    return train_losses, test_accuracies, best_model_path

# 损失景观可视化
def compute_loss_landscape(model, loader, criterion, device, learning_rates, steps=100):
    model.eval()
    max_curve = []
    min_curve = []
    original_state = {name: param.clone() for name, param in model.named_parameters()}

    for step in range(steps):
        max_loss = float('-inf')
        min_loss = float('inf')
        for lr in learning_rates:
            # 重置模型参数
            for name, param in model.named_parameters():
                param.data = original_state[name].clone()
            
            # 计算梯度
            model.zero_grad()
            inputs, labels = next(iter(loader))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 沿着梯度方向移动
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.add_(-lr * param.grad)
            
            # 计算新损失
            outputs = model(inputs)
            new_loss = criterion(outputs, labels).item()
            max_loss = max(max_loss, new_loss)
            min_loss = min(min_loss, new_loss)

        max_curve.append(max_loss)
        min_curve.append(min_loss)

        # 更新参数以模拟训练步骤
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.data = original_state[name].clone()
                param.add_(-learning_rates[0] * param.grad)  # 使用第一个学习率更新
                original_state[name].data = param.data.clone()

    return max_curve, min_curve

# 可视化损失景观
def plot_loss_landscape(max_curve_res_bn, min_curve_res_bn, max_curve_no_bn, min_curve_no_bn, title="Loss Landscape"):
    plt.figure(figsize=(10, 6))
    steps = range(len(max_curve_res_bn))
    plt.plot(steps, max_curve_res_bn, 'b-', label='Max Loss (Res+BN)')
    plt.plot(steps, min_curve_res_bn, 'b--', label='Min Loss (Res+BN)')
    plt.fill_between(steps, min_curve_res_bn, max_curve_res_bn, alpha=0.2, color='blue')
    plt.plot(steps, max_curve_no_bn, 'r-', label='Max Loss (No BN)')
    plt.plot(steps, min_curve_no_bn, 'r--', label='Min Loss (No BN)')
    plt.fill_between(steps, min_curve_no_bn, max_curve_no_bn, alpha=0.2, color='red')
    plt.title(title)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_landscape_res_bn.png')
    plt.close()

# 主函数
def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载CIFAR-10数据集
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # 初始化模型
    model_res_bn = VGG_A_Res_BN(inp_ch=3, num_classes=10).to(device)
    model_no_bn = VGG_A(inp_ch=3, num_classes=10).to(device)
    print(f"VGG-A with Residual Blocks and BN parameters: {get_number_of_parameters(model_res_bn)}")
    print(f"VGG-A without BN parameters: {get_number_of_parameters(model_no_bn)}")

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器实验
    optimizers = [
        ('SGD', optim.SGD(model_res_bn.parameters(), lr=0.01, momentum=0.9)),
        ('Adam', optim.Adam(model_res_bn.parameters(), lr=0.001))
    ]

    # 训练并比较优化器
    results = {}
    for opt_name, optimizer in optimizers:
        print(f"\nTraining VGG-A with Residual Blocks and BN using {opt_name}")
        train_losses, test_accuracies, model_path = train_model(
            model_res_bn, train_loader, test_loader, optimizer, criterion, device, epochs=50, model_name=f"vgg_a_res_bn_{opt_name.lower()}"
        )
        results[opt_name] = {'losses': train_losses, 'accuracies': test_accuracies, 'model_path': model_path}

    # 训练不带BN的模型（使用SGD）
    print("\nTraining VGG-A without BN using SGD")
    optimizer_no_bn = optim.SGD(model_no_bn.parameters(), lr=0.01, momentum=0.9)
    train_losses_no_bn, test_accuracies_no_bn, model_path_no_bn = train_model(
        model_no_bn, train_loader, test_loader, optimizer_no_bn, criterion, device, epochs=50, model_name="vgg_a_no_bn"
    )

    # 绘制训练损失和测试准确率
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for opt_name, result in results.items():
        plt.plot(result['losses'], label=f"Res+BN {opt_name}")
    plt.plot(train_losses_no_bn, label="No BN SGD")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for opt_name, result in results.items():
        plt.plot(result['accuracies'], label=f"Res+BN {opt_name}")
    plt.plot(test_accuracies_no_bn, label="No BN SGD")
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig('training_comparison_res_bn.png')
    plt.close()

    # 损失景观分析
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
    print("\nComputing loss landscape for VGG-A with Residual Blocks and BN")
    max_curve_res_bn, min_curve_res_bn = compute_loss_landscape(model_res_bn, train_loader, criterion, device, learning_rates)
    print("Computing loss landscape for VGG-A without BN")
    max_curve_no_bn, min_curve_no_bn = compute_loss_landscape(model_no_bn, train_loader, criterion, device, learning_rates)
    plot_loss_landscape(max_curve_res_bn, min_curve_res_bn, max_curve_no_bn, min_curve_no_bn, "Loss Landscape Comparison (Res+BN vs No BN)")

    # 保存数据集和模型权重链接（示例）
    dataset_link = "https://www.cs.toronto.edu/~kriz/cifar.html"  # 实际应上传至Google Drive
    model_weights_links = {opt_name: result['model_path'] for opt_name, result in results.items()}
    model_weights_links['No BN'] = model_path_no_bn
    print("\nDataset link:", dataset_link)
    print("Model weights saved at:", model_weights_links)

if __name__ == '__main__':
    main()
