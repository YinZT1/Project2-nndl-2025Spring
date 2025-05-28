### task1 保姆级教学

#### 评估


首先从网盘中下载给定的权重链接，将文件放在weight文件夹下。

运行`python evaluation.py --checkpoint_path ./weight/<name>`即可

#### 训练
脚本支持以下超参数：

模型结构 (Filters/Neurons):
- standard: 假设是您代码中 VGG16 的标准 filter 和全连接层配置。
- light: filter 数量减半，全连接层神经元数量减少 (例如从 4096 -> 1024)。

激活函数:
- relu: nn.ReLU
- leaky_relu: nn.LeakyReLU(0.1)
- tanh: nn.Tanh

损失函数:
- cross_entropy: nn.CrossEntropyLoss
- nll_loss: nn.NLLLoss (需要配合模型末尾的 LogSoftmax)

Optimizer:
- adam: optim.Adam (例如 lr=0.001)
- sgd: optim.SGD (例如 lr=0.01, momentum=0.9)

学习率lr

动量momentum

训练回合epochs

weight_decay权重衰减系数

`run2.sh`中已经提供了一份初始脚本，你可以修改参数跑。**tips:注意你需要先把CIFAR10数据集放入./data中。**