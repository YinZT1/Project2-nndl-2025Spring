# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # 修正导入路径以匹配项目结构 (task1/utils/nn.py)
# from utils.nn import init_weights_

# # 辅助函数：创建激活层实例
# def _get_activation_layer(activation_fn_class, inplace_default=True):
#     """
#     实例化激活函数。如果可能且指定，尝试使用inplace=True。
#     """
#     if inplace_default:
#         try:
#             # 尝试为ReLU等支持inplace的激活函数设置inplace=True
#             return activation_fn_class(inplace=True)
#         except TypeError:
#             # 如果激活函数不支持inplace参数，或者inplace=True导致错误
#             return activation_fn_class()
#     else:
#         return activation_fn_class()

# class ResidualBlock(nn.Module):
#     """
#     A residual block with two convolutional layers, BatchNorm, and a configurable activation function.
#     Includes optional downsampling via stride or 1x1 convolution.
#     """
#     expansion = 1 # For basic blocks, input channels == output channels

#     def __init__(self, in_channels, out_channels, stride=1, activation_fn_class=nn.ReLU):
#         super(ResidualBlock, self).__init__()
#         # Main path
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.activation = _get_activation_layer(activation_fn_class)
#         self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

#         # Skip connection (shortcut)
#         self.shortcut = nn.Sequential()
#         # If stride > 1 (downsampling) or in_channels != out_channels*expansion,
#         # we need to project the shortcut connection.
#         if stride != 1 or in_channels != out_channels * self.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * self.expansion)
#             )

#     def forward(self, x):
#         identity = x # Store the input for the shortcut

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.activation(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         out += self.shortcut(identity) # Add the shortcut connection
#         out = self.activation(out) # Final activation after addition
#         return out

# class ResVGG16(nn.Module):
#     """
#     A VGG16-inspired model using Residual Blocks, adapted for CIFAR-10 (32x32 input).
#     Includes BatchNorm (within blocks) and Dropout (in classifier).
#     Allows configuration of activation functions, filter numbers, and classifier neuron numbers.
#     """
#     def __init__(self, block=ResidualBlock, 
#                  num_blocks_list=[2, 2, 3, 3, 3], 
#                  num_classes=10, 
#                  dropout_p=0.5,
#                  activation_fn_class=nn.ReLU,
#                  initial_conv_out_channels=64,
#                  channels_list=[64, 128, 256, 512, 512],
#                  classifier_hidden_neurons_list=[512, 512]):
#         super(ResVGG16, self).__init__()

#         if len(channels_list) != len(num_blocks_list):
#             raise ValueError("channels_list must have the same length as num_blocks_list.")

#         self.activation_fn_class = activation_fn_class
#         self.in_channels = initial_conv_out_channels

#         # Initial convolution - Input 3x32x32
#         self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.in_channels)
#         self.initial_activation = _get_activation_layer(self.activation_fn_class)
#         # After conv1: initial_conv_out_channels x32x32

#         # VGG-like stages using Residual Blocks
#         self.layer1 = self._make_layer(block, channels_list[0], num_blocks_list[0], stride=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                      

#         self.layer2 = self._make_layer(block, channels_list[1], num_blocks_list[1], stride=1) 
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                      

#         self.layer3 = self._make_layer(block, channels_list[2], num_blocks_list[2], stride=1) 
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)                      

#         self.layer4 = self._make_layer(block, channels_list[3], num_blocks_list[3], stride=1) 
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)                      

#         self.layer5 = self._make_layer(block, channels_list[4], num_blocks_list[4], stride=1) 
#         self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)                      

#         # Adaptive pooling to ensure 1x1 spatial size before FC layer
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#         # Classifier with Dropout
#         classifier_layers = []
#         # Input features to the first FC layer: output channels of the last residual stage * expansion factor
#         fc_input_features = channels_list[-1] * block.expansion
        
#         classifier_layers.append(nn.Flatten())

#         for num_neurons in classifier_hidden_neurons_list:
#             classifier_layers.append(nn.Linear(fc_input_features, num_neurons))
#             classifier_layers.append(_get_activation_layer(self.activation_fn_class))
#             classifier_layers.append(nn.Dropout(p=dropout_p))
#             fc_input_features = num_neurons # Input for the next FC layer

#         classifier_layers.append(nn.Linear(fc_input_features, num_classes))
#         self.classifier = nn.Sequential(*classifier_layers)

#         # Initialize weights
#         self.apply(init_weights_) 

#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         # The first block in a layer might handle downsampling (stride > 1)
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for current_stride in strides:
#             layers.append(block(self.in_channels, out_channels, current_stride, 
#                                 activation_fn_class=self.activation_fn_class))
#             self.in_channels = out_channels * block.expansion # Update in_channels for the next block/layer
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.initial_activation(self.bn1(self.conv1(x)))
        
#         out = self.pool1(self.layer1(out))
#         out = self.pool2(self.layer2(out))
#         out = self.pool3(self.layer3(out))
#         out = self.pool4(self.layer4(out))
#         out = self.pool5(self.layer5(out))
        
#         out = self.avgpool(out)
#         out = self.classifier(out)
#         return out

# task1/models/vgg.py (示例修改)
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义不同激活函数的辅助函数
def get_activation(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'leaky_relu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif name == 'tanh':
        return nn.Tanh()
    # 可以添加更多...
    else:
        raise ValueError(f"Unsupported activation: {name}")

cfg_arch = {
    'VGG16_standard': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16_light': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    # 可以添加更多cfg
}

fc_layers_cfg = {
    'standard': (512, [4096, 4096]), # (input_multiplier_from_last_conv_filter, [fc_hidden_dims])
    'light': (256, [1024, 512]),
}


class VGG(nn.Module):
    def __init__(self, vgg_name='VGG16_standard', activation_name='relu', num_classes=10, init_weights=True, use_log_softmax=False):
        super(VGG, self).__init__()
        self.vgg_name = vgg_name
        self.activation_name = activation_name
        self.use_log_softmax = use_log_softmax # 新增参数

        if vgg_name not in cfg_arch:
            raise ValueError(f"Unsupported VGG name: {vgg_name}")
        current_cfg = cfg_arch[vgg_name]

        self.features = self._make_layers(current_cfg, activation_name)

        # 确定分类器的输入维度
        # 假设CIFAR10输入32x32，经过5个'M'（池化使尺寸减半），特征图为1x1
        # 输入特征数 = 最后一个卷积层的filter数 * 1 * 1
        last_conv_filters = 0
        for x in reversed(current_cfg):
            if isinstance(x, int):
                last_conv_filters = x
                break
        
        # 根据 vgg_name 选择合适的 fc_config
        fc_config_name = 'standard' if 'standard' in vgg_name else 'light' # 简单映射逻辑
        if fc_config_name not in fc_layers_cfg:
             fc_config_name = 'standard' # 默认
        
        fc_input_multiplier, fc_hidden_dims = fc_layers_cfg[fc_config_name]
        
        # 如果实际的 last_conv_filters 与 fc_config 中的 input_multiplier 不同，优先使用实际计算的
        classifier_input_dim = last_conv_filters 

        classifier_layers = []
        in_dim = classifier_input_dim
        for hidden_dim in fc_hidden_dims:
            classifier_layers.append(nn.Linear(in_dim, hidden_dim))
            classifier_layers.append(get_activation(activation_name))
            classifier_layers.append(nn.Dropout())
            in_dim = hidden_dim
        classifier_layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.use_log_softmax: # 根据参数决定是否使用 LogSoftmax
            return F.log_softmax(x, dim=1)
        return x

    def _make_layers(self, cfg, activation_name, batch_norm=False): # 假设您的ResVGG16不需要BN，或已内置
        layers = []
        in_channels = 3
        activation_func = get_activation(activation_name)
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm: # 如果您的模型支持batch_norm
                    layers += [conv2d, nn.BatchNorm2d(v), activation_func]
                else:
                    layers += [conv2d, activation_func]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu' if 'relu' in self.activation_name else 'tanh') # 简单处理
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)