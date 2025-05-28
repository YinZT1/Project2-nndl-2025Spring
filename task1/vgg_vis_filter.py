import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# 假设您的 VGG 模型定义在 models.vgg16 模块中
# from models.vgg16 import VGG # 确保这个导入路径根据您的项目结构是正确的
# 下面这行是根据您之前提供的newvgg16.py的导入方式
from models.vgg16 import VGG


def load_model_for_visualization(checkpoint_path, vgg_config_name, activation_name, num_classes=10):
    """加载模型用于可视化"""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    # 实例化模型结构 (与训练时一致)
    # 注意: 如果损失函数是nll_loss, use_log_softmax应为True, 但可视化滤波器时通常不需要这个
    model = VGG(vgg_name=vgg_config_name,
                activation_name=activation_name,
                num_classes=num_classes,
                use_log_softmax=False, # 对于权重可视化，通常不需要
                init_weights=False) # 从检查点加载权重，不需要重新初始化

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None

    # 加载检查点
    # checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) # 加载到CPU
    # 兼容Colab/GPU环境，如果GPU可用则加载到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 为了处理可能的 DataParallel 包装 (如果模型是在多个GPU上训练并保存的)
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k # 移除'module.'前缀
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval() # 设置为评估模式
    print("Model loaded successfully.")
    return model

def normalize_filter(f):
    """将滤波器权重归一化到0-1范围"""
    f_min, f_max = f.min(), f.max()
    return (f - f_min) / (f_max - f_min + 1e-6) # 防止除以零

def visualize_conv_filters(model, layer_index=0, max_filters_to_show=16):
    """
    可视化指定卷积层的滤波器。
    Args:
        model: 加载好的PyTorch模型。
        layer_index (int): model.features 中卷积层的索引。
                           例如，0 代表第一个卷积层，如果后面紧跟激活层，则下一个卷积层索引可能不是1。
                           您需要根据 print(model) 的输出来确定准确的索引。
        max_filters_to_show (int): 最多显示多少个滤波器。
    """
    # --- 找到目标卷积层 ---
    target_layer = None
    current_conv_idx = 0
    target_layer_name = ""

    # 遍历 features 模块寻找第 layer_index 个 Conv2d 层
    # (注意：这里的 layer_index 指的是第几个 Conv2d 层，而不是 nn.Sequential 中的绝对索引)
    conv_layers_found = 0
    for name, layer in model.features.named_children():
        if isinstance(layer, torch.nn.Conv2d):
            if conv_layers_found == layer_index:
                target_layer = layer
                target_layer_name = f"features.{name}"
                break
            conv_layers_found += 1
    
    if target_layer is None:
        print(f"Error: Conv2d layer at effective index {layer_index} not found in model.features.")
        print("Available layers in model.features:")
        for name, layer in model.features.named_children():
            print(f"  {name}: {layer.__class__.__name__}")
        return

    print(f"Visualizing filters from layer: {target_layer_name} ({target_layer})")
    
    weights = target_layer.weight.data.clone().cpu() # (out_channels, in_channels, H, W)
    
    if weights.shape[1] != 3 and weights.shape[1] != 1:
        print(f"Visualizing filters from deeper layers (in_channels={weights.shape[1]}) is more complex.")
        print("This function is best for the first layer (in_channels=3) or layers with in_channels=1.")
        # 对于更深层，可以考虑可视化每个输出通道的第一个输入通道对应的权重
        weights = weights[:, 0:1, :, :] # 取第一个输入通道的权重进行可视化
        # 或者对所有输入通道的权重取平均/最大值等，但这会丢失信息

    num_filters = weights.shape[0]
    num_to_show = min(num_filters, max_filters_to_show)

    # 计算网格布局
    num_cols = int(np.sqrt(num_to_show))
    num_rows = int(np.ceil(num_to_show / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axes = axes.flatten() # 将二维数组展平，方便索引

    for i in range(num_to_show):
        filt = weights[i] # (in_channels, H, W)
        
        # 如果是第一个卷积层 (in_channels=3)，则 filt 是 (3, H, W)
        # 如果是更深层且我们只取了第一个输入通道，则是 (1, H, W)
        if filt.shape[0] == 3: # RGB filter
            # PyTorch 张量是 (C, H, W), Matplotlib imshow 期望 (H, W, C) or (H,W)
            filt_display = normalize_filter(filt).permute(1, 2, 0).numpy()
        elif filt.shape[0] == 1: # Grayscale filter
            filt_display = normalize_filter(filt.squeeze()).numpy() # (H,W)
        else:
            # 对于其他情况，简单显示第一个通道
            filt_display = normalize_filter(filt[0,:,:]).numpy()

        ax = axes[i]
        if filt.shape[0] == 1 or (filt_display.ndim == 2 and filt_display.shape[-1] !=3) : # Grayscale
            ax.imshow(filt_display, cmap='gray')
        else: # RGB
            ax.imshow(filt_display)
        ax.set_title(f'Filter {i+1}')
        ax.axis('off')

    # 隐藏多余的子图
    for j in range(num_to_show, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Filters from {target_layer_name} (First {num_to_show} of {num_filters})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # rect 调整总标题的位置
    plt.savefig(r"/remote-home/yinzhitao/pj2/codes/VGG_BatchNorm/task1_train_fig")

if __name__ == '__main__':
    # --- 示例：如何使用 ---
    # 1. 首先确定您的最佳模型检查点路径和配置
    # 例如，根据您的结果，Run 6 表现很好
    example_checkpoint_path = "/remote-home/yinzhitao/pj2/codes/VGG_BatchNorm/grid_search_results/run_7_VGG16_standard_leaky_relu_nll_loss_adam/run_7_VGG16_standard_leaky_relu_nll_loss_adam_vgg_VGG16_standard_act_leaky_relu_loss_nll_loss_opt_adam_lr_0.0001_best.pth.tar"
    example_vgg_config = "VGG16_standard"
    example_activation = "leaky_relu"
    

    if not os.path.exists(example_checkpoint_path) or "run_X" in example_checkpoint_path:
        print("="*50)
        print("注意: 您需要将下面的 'example_checkpoint_path', 'example_vgg_config', 和 'example_activation' \n"
              "替换为您实际的最佳模型检查点路径和对应的配置才能运行此示例。")
        print("例如，从您的实验结果中选择一个准确率高的模型。")
        print("如：'./grid_search_results/run_6_VGG16_standard_leaky_relu_cross_entropy_sgd/run_6_VGG16_standard_leaky_relu_cross_entropy_sgd_best.pth.tar'")
        print("并设置 vgg_config='VGG16_standard', activation='leaky_relu'")
        print("="*50)
    else:
        model = load_model_for_visualization(example_checkpoint_path, example_vgg_config, example_activation)
        if model:
            # 打印模型结构以确定卷积层索引
            print("\nModel Structure (Features):")
            for name, layer in model.features.named_children():
                print(f"  {name}: {layer}")
            print("-" * 30)
            
            # 可视化第一个卷积层 (通常是 model.features[0] 如果它是Conv2d)
            # layer_index=0 指的是模型中第 0 个 (即第一个) Conv2d 层
            visualize_conv_filters(model, layer_index=0, max_filters_to_show=16) 
            
            # 如果想可视化更深层的卷积核，例如第2个卷积层
            # (您需要根据上面打印出的模型结构确定其在 nn.Sequential 中的确切索引或名称)
            # visualize_conv_filters(model, layer_index=1, max_filters_to_show=16)