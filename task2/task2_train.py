import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from uuid import uuid4
from models.vgg import VGG_A

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

# VGG-A模型（带Batch Normalization）
class VGG_A_BN(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 5
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

            # 沿着梯度方向移动（避免原地操作）
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # 使用 torch.add 替代 param.add_
                        param.data = torch.add(param.data, param.grad, alpha=-lr)
            
            # 计算新损失
            outputs = model(inputs)
            new_loss = criterion(outputs, labels).item()
            max_loss = max(max_loss, new_loss)
            min_loss = min(min_loss, new_loss)

        max_curve.append(max_loss)
        min_curve.append(min_loss)

        # 更新参数以模拟训练步骤（避免原地操作）
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # 使用 torch.add 更新参数并保存状态
                    param.data = torch.add(original_state[name].clone(), param.grad, alpha=-learning_rates[0])
                    original_state[name].data = param.data.clone()

    return max_curve, min_curve

# 可视化损失景观
def plot_loss_landscape(max_curve_bn, min_curve_bn, max_curve_no_bn, min_curve_no_bn, title="Loss Landscape"):
    plt.figure(figsize=(10, 6))
    steps = range(len(max_curve_bn))
    plt.plot(steps, max_curve_bn, 'b-', label='Max Loss (BN)')
    plt.plot(steps, min_curve_bn, 'b--', label='Min Loss (BN)')
    plt.fill_between(steps, min_curve_bn, max_curve_bn, alpha=0.2, color='blue')
    plt.plot(steps, max_curve_no_bn, 'r-', label='Max Loss (No BN)')
    plt.plot(steps, min_curve_no_bn, 'r--', label='Min Loss (No BN)')
    plt.fill_between(steps, min_curve_no_bn, max_curve_no_bn, alpha=0.2, color='red')
    plt.title(title)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_landscape.png')
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
    model_bn = VGG_A_BN(inp_ch=3, num_classes=10).to(device)
    model_no_bn = VGG_A(inp_ch=3, num_classes=10).to(device)  # 使用原始VGG_A
    model_bn = torch.load(r"/remote-home/yinzhitao/pj2/codes/VGG_BatchNorm/vgg_a_bn_adam_best.pth")
    model_no_bn = torch.load(r"/remote-home/yinzhitao/pj2/codes/VGG_BatchNorm/vgg_a_no_bn_best.pth")
    print(f"VGG-A with BN parameters: {get_number_of_parameters(model_bn)}")
    print(f"VGG-A without BN parameters: {get_number_of_parameters(model_no_bn)}")

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # # 优化器实验
    # optimizers = [
    #     ('SGD', optim.SGD(model_bn.parameters(), lr=0.01, momentum=0.9)),
    #     ('Adam', optim.Adam(model_bn.parameters(), lr=0.001))
    # ]

    # # 训练并比较优化器
    # results = {}
    # for opt_name, optimizer in optimizers:
    #     print(f"\nTraining VGG-A with BN using {opt_name}")
    #     train_losses, test_accuracies, model_path = train_model(
    #         model_bn, train_loader, test_loader, optimizer, criterion, device, epochs=50, model_name=f"vgg_a_bn_{opt_name.lower()}"
    #     )
    #     results[opt_name] = {'losses': train_losses, 'accuracies': test_accuracies, 'model_path': model_path}

    # # 训练不带BN的模型（使用SGD）
    # print("\nTraining VGG-A without BN using SGD")
    # optimizer_no_bn = optim.SGD(model_no_bn.parameters(), lr=0.01, momentum=0.9)
    # train_losses_no_bn, test_accuracies_no_bn, model_path_no_bn = train_model(
    #     model_no_bn, train_loader, test_loader, optimizer_no_bn, criterion, device, epochs=50, model_name="vgg_a_no_bn"
    # )

    # # 绘制训练损失和测试准确率
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # for opt_name, result in results.items():
    #     plt.plot(result['losses'], label=f"BN {opt_name}")
    # plt.plot(train_losses_no_bn, label="No BN SGD")
    # plt.title("Training Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.grid(True)

    # plt.subplot(1, 2, 2)
    # for opt_name, result in results.items():
    #     plt.plot(result['accuracies'], label=f"BN {opt_name}")
    # plt.plot(test_accuracies_no_bn, label="No BN SGD")
    # plt.title("Test Accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy (%)")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('training_comparison.png')
    # plt.close()

    # 损失景观分析
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
    print("\nComputing loss landscape for VGG-A with BN")
    max_curve_bn, min_curve_bn = compute_loss_landscape(model_bn, train_loader, criterion, device, learning_rates)
    print("Computing loss landscape for VGG-A without BN")
    max_curve_no_bn, min_curve_no_bn = compute_loss_landscape(model_no_bn, train_loader, criterion, device, learning_rates)
    plot_loss_landscape(max_curve_bn, min_curve_bn, max_curve_no_bn, min_curve_no_bn, "Loss Landscape Comparison")

    print("Model weights saved at:", model_weights_links)

if __name__ == '__main__':
    main()