# CentralFlows Mini-batch 修改说明

## 修改概述

本修改为Central Flows项目添加了mini-batch训练支持，并扩展支持多进程同时优化对比。现在支持在一个实验中同时运行full-batch GD、mini-batch SGD和central flow，提供了完整的优化算法理论对比平台。

## 主要修改

### 1. 数据集修改 (`src/datasets.py`)
- 添加 `batch_size` 参数控制mini-batch大小
- 修改 `Dataset` NamedTuple包含完整数据集和batch数据集
- 自动计算和创建mini-batches用于训练

### 2. 主训练脚本修改 (`main.py`)
- 新增 `--compare-full-vs-mini` 参数启用batch级别对比
- 新增 `--batch-size` 参数设置mini-batch大小
- 支持 `--runs discrete central` 同时运行discrete和central进程
- 自动计算正确的 `total_steps = epochs * batches * processes`
- 实现三进程同步训练：full-batch、mini-batch、central flow

### 3. 可视化脚本增强 (`plot_training_curves.py`)
- 增强错误处理，防御性编程防止数组越界
- 支持三进程数据同时可视化
- 添加调试信息显示HDF5文件结构
- 自动检测可用的数据类型并显示

### 4. 存储空间自动适配
- 动态计算HDF5数组大小，容纳所有进程的所有步数
- 避免IndexError，支持任意组合的进程数量

## 使用方法

### 1. 基础mini-batch对比

```bash
python main.py opt:gd data:moons arch:mlp \
    --opt.lr=0.02 \
    --epochs=20 \
    --batch-size=32 \
    --compare-full-vs-mini \
    --runs discrete
```

这会运行full-batch GD vs mini-batch SGD对比，其中mini-batch使用batch_size=32。

### 2. 三进程完整对比

```bash
python main.py opt:gd data:moons arch:mlp \
    --opt.lr=0.02 \
    --epochs=20 \
    --batch-size=32 \
    --compare-full-vs-mini \
    --runs discrete central
```

这会同时运行：
- full-batch discrete GD
- mini-batch discrete SGD
- central flow (连续时间优化)

### 3. 单进程训练 (原始模式)

```bash
python main.py opt:gd data:cifar10 arch:cnn \
    --opt.lr=0.01 \
    --epochs=50 \
    --batch-size=128 \
    --compare-full-vs-mini=False \
    --runs discrete
```

## 数据保存格式

### HDF5结构
```
experiments/{exp_id}/data.hdf5
├── step/                    # 索引数组 (辅助用)
├── discrete_full/           # full-batch进程数据
│   ├── loss                 # [N]数组 - 每个batch的loss值
│   ├── grad_norm           # [N]数组 - 每个batch的梯度范数
│   └── hessian_eigs        # [N, K]数组 - 前K个Hessian特征值
├── discrete_mini/           # mini-batch进程数据
│   ├── loss
│   ├── grad_norm
│   └── hessian_eigs
└── central/                 # central flow进程数据
    ├── loss
    ├── grad_norm
    └── hessian_eigs
```

**数组长度N**: `epochs × num_batches × num_processes`

**示例计算**: 如果 `epochs=20, batch_size=32, runs=discrete central`，
- 数据集被分7个batch: `(100÷32) = ~4个完整batch`
- 进程数: `discrete`创建2个 + `central`1个 = `3个进程`
- 总步数: `20 × 7 × 3 = 420` (**因数据集大小而异**)

**total_steps 用于**: 初始化HDF5数组存储空间
```python
# 在DataSaver.__init__() 中使用:
with DataSaver(folder / "data.hdf5", 0, total_steps) as data_saver
```

## 可视化工具

运行可视化脚本查看完整对比：

```bash
# 接下来显示图片（GUI环境）
python plot_training_curves.py experiments/your_exp_id

# 保存图片到文件（服务器推荐）
python plot_training_curves.py experiments/your_exp_id --save-path results.png --no-show

# 保存为PDF等其他格式
python plot_training_curves.py experiments/your_exp_id --save-path results.pdf --no-show
```

### 可视化输出
生成的图包含4个子图：

1. **训练损失曲线** - 三个进程的loss下降轨迹
2. **梯度范数曲线** - 显示梯度收敛行为
3. **Hessian特征值** - 显示sharpness演化（如果有数据）
4. **训练总结** - 显示最终loss和统计对比
